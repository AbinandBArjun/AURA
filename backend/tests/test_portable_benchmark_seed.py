from copy import deepcopy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx

import pytest
from qdrant_client import models
from sqlalchemy import func, select

from app.models import Document, DocumentChunk
from evaluation.portable_benchmark import seed as seeder
from evaluation.portable_benchmark.corpus import CORPUS_DIR
from evaluation.portable_benchmark.seed_assets import prepare_corpus
from evaluation.portable_benchmark.seed_infra import (
    COLLECTION, COMPOSE_FILE, DATABASE, POSTGRES_IMAGE, ROLE, ResourceIdentity, SeedError,
    Targets, resolve_targets, validate_container, validate_request,
)
from evaluation.portable_benchmark.seed_record import SeedRecord
from evaluation.portable_benchmark.seed_transport import ContainerTransportError, ContainerQdrantTransport
from evaluation.portable_benchmark.seed_vectors import DIMENSIONS, PAYLOAD_INDEXES, create_vectors, point_digests


PROJECT = "aura-benchmark-tests"
POINT = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def postgres_inspect():
    mapping = {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "15432"}]}
    return {
        "Id": "a" * 64,
        "Config": {
            "Env": [
                f"POSTGRES_DB={DATABASE}", f"POSTGRES_USER={ROLE}", f"POSTGRES_PASSWORD={'a' * 64}",
                "PGDATA=/var/lib/postgresql/data/pgdata",
            ],
            "Image": POSTGRES_IMAGE,
            "Labels": {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.service": "postgres",
                "com.docker.compose.oneoff": "False",
                "com.docker.compose.project.config_files": str(COMPOSE_FILE),
                "org.aura.portable-benchmark": "1",
            },
        },
        "State": {"Running": True},
        "HostConfig": {
            "NetworkMode": f"{PROJECT}_default",
            "Tmpfs": {"/var/lib/postgresql/data": ""},
            "PortBindings": deepcopy(mapping),
        },
        "NetworkSettings": {"Networks": {f"{PROJECT}_default": {}}, "Ports": mapping},
        "Mounts": [],
    }


@pytest.mark.parametrize(
    ("context", "project", "pg_port", "vector_port"),
    [
        ("", PROJECT, 15432, 16333),
        ("local", "aura", 15432, 16333),
        ("local", PROJECT, 5433, 16333),
        ("local", PROJECT, 15432, 6333),
        ("local", PROJECT, 15432, 15432),
        ("local", PROJECT, 65536, 16333),
    ],
)
def test_unsafe_target_arguments_are_rejected(context, project, pg_port, vector_port):
    with pytest.raises(SeedError):
        validate_request(context, project, pg_port, vector_port)


def test_remote_docker_context_is_rejected(monkeypatch):
    docker = Mock(return_value='[{"Endpoints":{"docker":{"Host":"tcp://example.invalid:2375"}}}]')
    monkeypatch.setattr("evaluation.portable_benchmark.seed_infra.docker", docker)
    with pytest.raises(SeedError, match="local daemon"):
        resolve_targets("remote", PROJECT, 15432, 16333)
    docker.assert_called_once_with("remote", "context", "inspect", "remote")


def test_project_bridge_with_an_unrelated_container_is_rejected(monkeypatch):
    pg_id, qdrant_id, unrelated_id = ("a" * 64, "b" * 64, "c" * 64)
    containers = [
        {"Id": identifier, "Config": {"Labels": {"com.docker.compose.service": service}}}
        for identifier, service in ((pg_id, "postgres"), (qdrant_id, "qdrant"))
    ]
    network = {
        "Driver": "bridge", "Scope": "local",
        "Labels": {"com.docker.compose.project": PROJECT, "com.docker.compose.network": "default"},
        "Containers": {identifier: {} for identifier in (pg_id, qdrant_id, unrelated_id)},
    }
    docker = Mock(side_effect=[
        '[{"Endpoints":{"docker":{"Host":"unix:///var/run/docker.sock"}}}]',
        f"{pg_id}\n{qdrant_id}\n", json.dumps(containers), json.dumps([network]),
    ])
    monkeypatch.setattr("evaluation.portable_benchmark.seed_infra.docker", docker)
    with pytest.raises(SeedError, match="exclusive local Compose bridge"):
        resolve_targets("local", PROJECT, 15432, 16333)
    assert docker.call_count == 4


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("Config", "Image"), "postgres:16"),
        (("Config", "Labels", "com.docker.compose.project.config_files"), "docker-compose.yml"),
        (("HostConfig", "Binds"), ["development:/var/lib/postgresql/data"]),
        (("HostConfig", "Tmpfs"), {}),
        (("NetworkSettings", "Ports"), {"5432/tcp": [{"HostIp": "0.0.0.0", "HostPort": "15432"}]}),
        (("Mounts",), [{"Type": "volume", "Destination": "/var/lib/postgresql/data"}]),
    ],
)
def test_container_proof_rejects_unsafe_configuration(postgres_inspect, path, value):
    assert validate_container(postgres_inspect, PROJECT, "postgres", 15432) == "a" * 64
    target = postgres_inspect
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    with pytest.raises(SeedError, match="Refusing unverified"):
        validate_container(postgres_inspect, PROJECT, "postgres", 15432)


@pytest.fixture
def identity():
    return ResourceIdentity(
        docker_context="local", project=PROJECT, postgres_container="a" * 64,
        qdrant_container="b" * 64, postgres_port=15432, qdrant_port=16333,
        postgres_system_id="123456789", database=DATABASE, collection=COLLECTION,
        execution_id="a" * 32, qdrant_instance="pid:[17]:104",
    )


def test_connection_rejects_different_postgres_identity(identity):
    connection = Mock()
    connection.exec_driver_sql.return_value.one.return_value = (DATABASE, ROLE, "999999999")
    targets = Targets(identity, "a" * 64, "b" * 64)
    with pytest.raises(SeedError, match="endpoint did not match"):
        targets.verify_connections(connection)


def test_unauthenticated_qdrant_is_rejected_before_sending_key(monkeypatch, identity):
    connection = Mock()
    connection.exec_driver_sql.return_value.one.return_value = (DATABASE, ROLE, identity.postgres_system_id)
    client = Mock()
    client.get.return_value.status_code = 200
    httpx_client = Mock()
    httpx_client.return_value.__enter__ = Mock(return_value=client)
    httpx_client.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("evaluation.portable_benchmark.seed_infra.httpx.Client", httpx_client)
    with pytest.raises(SeedError, match="did not require authentication"):
        Targets(identity, "a" * 64, "b" * 64).verify_connections(connection)
    client.get.assert_called_once_with("http://127.0.0.1:6333/collections")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return prepare_corpus(CORPUS_DIR, tmp_path_factory.mktemp("seed-pdfs") / "pdfs")


@pytest.mark.parametrize("mutation", ["missing", "altered", "wrong_passage"])
def test_corpus_defects_rejected_before_pdf_or_sql_writes(tmp_path, mutation):
    bundle = tmp_path / "bundle"
    shutil.copytree(CORPUS_DIR / "sources", bundle / "sources")
    for name in ("manifest.json", "dataset.json", "passage_ids.json"):
        shutil.copyfile(CORPUS_DIR / name, bundle / name)
    source = bundle / "sources" / "backup-assurance.txt"
    if mutation == "missing":
        source.unlink()
        message = "Missing source"
    elif mutation == "altered":
        source.write_text("Altered synthetic test text\n")
        message = "SHA-256 mismatch"
    else:
        references = bundle / "passage_ids.json"
        value = references.read_text()
        references.write_text(value.replace("p1:backup-assurance:0:", "p1:backup-assurance:99:", 1))
        message = "Passage identities"
    with pytest.raises(ValueError, match=message):
        prepare_corpus(bundle, tmp_path / "pdfs")
    assert not (tmp_path / "pdfs").exists()


def test_real_pdf_path_produces_all_reviewed_passages(prepared):
    assert len(prepared.documents) == 8
    assert sum(len(doc.passages) for doc in prepared.documents) == 40
    assert all(doc.pages and doc.pdf_bytes > 0 for doc in prepared.documents)


def test_extracted_passage_identity_mismatch_is_rejected(monkeypatch, tmp_path):
    from evaluation.portable_benchmark import seed_assets
    extract = seed_assets.extract_pdf_text

    def alter(path):
        result = extract(path)
        result["pages"][0]["text"] += "\nA different sentence added after extraction."
        return result

    monkeypatch.setattr(seed_assets, "extract_pdf_text", alter)
    with pytest.raises(SeedError, match="Passage identities"):
        prepare_corpus(CORPUS_DIR, tmp_path / "pdfs")


def test_sql_mapping_detects_altered_chunk(db_session, prepared, identity):
    seeder.insert_corpus(db_session, prepared)
    _, _, documents, points = seeder.read_corpus(db_session, prepared, identity)
    assert len(documents) == 8
    assert len(points) == sum(p.chunk.retrievable for d in prepared.documents for p in d.passages)
    assert len({p["chunk_id"] for p in points.values()}) == len(points)
    assert db_session.scalar(select(func.count()).select_from(DocumentChunk)) == 40
    chunk = db_session.scalars(select(DocumentChunk)).first()
    chunk.text = "Modified passage"
    with pytest.raises(SeedError, match="SQL chunk differs"):
        seeder.read_corpus(db_session, prepared, identity)


@pytest.fixture
def record_data(prepared, identity):
    return {
        "schema_version": 1, "corpus_fingerprint": prepared.fingerprint, "corpus_version": prepared.corpus_version,
        "extraction_protocol": prepared.extraction, "chunking_protocol": prepared.chunking,
        "protocol_fingerprint": prepared.protocol, "passage_id_scheme": "p1", "alembic_revision": "seed-revision",
        "model_id": "minilm", "model_revision": "a" * 40, "model_manifest_sha256": "a" * 64,
        "vector_dimensions": DIMENSIONS, "resources": identity.model_dump(), "owner_id": 1, "workspace_id": 1,
        "documents": [{
            "document_key": prepared.documents[0].key, "document_id": 1, "passages": [{
                "passage_id": prepared.documents[0].passages[0].passage_id, "chunk_id": 1,
                "vector_id": POINT, "vector_sha256": "b" * 64,
            }],
        }],
    }


def test_record_round_trips_without_credentials(record_data):
    record = SeedRecord.model_validate(record_data)
    assert record.model_dump() == record_data
    assert SeedRecord.model_validate_json(record.model_dump_json()) == record
    assert "password" not in record.model_dump_json() and "api_key" not in record.model_dump_json()


@pytest.mark.parametrize(("key", "value"), [("schema_version", 2), ("owner_id", True), ("vector_dimensions", 128)])
def test_readiness_schema_rejects_invalid_fields(record_data, key, value):
    record_data[key] = value
    with pytest.raises(ValueError, match=key):
        SeedRecord.model_validate(record_data)


@pytest.mark.parametrize("duplicate", ["chunk_id", "vector_id"])
def test_duplicate_runtime_mapping_is_rejected(record_data, prepared, duplicate):
    passages = record_data["documents"][0]["passages"]
    another = {
        "passage_id": prepared.documents[0].passages[1].passage_id, "chunk_id": 2,
        "vector_id": "00000000-0000-0000-0000-000000000002", "vector_sha256": "b" * 64,
    }
    another[duplicate] = passages[0][duplicate]
    passages.append(another)
    with pytest.raises(ValueError, match="Duplicate"):
        SeedRecord.model_validate(record_data)


@pytest.fixture
def vectors():
    payloads = {POINT: {"text": "test", "workspace_id": 1}}
    original = [1.0] + [0.0] * (DIMENSIONS - 1)
    client = Mock()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(
            vectors=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
        )),
        payload_schema={name: SimpleNamespace(data_type=kind) for name, kind in PAYLOAD_INDEXES.items()},
    )
    point = SimpleNamespace(id=POINT, payload=payloads[POINT].copy(), vector=original.copy())
    client.scroll.return_value = ([point], None)
    return client, payloads, point, original


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "stale", "short", "changed_values"])
def test_invalid_vector_state_fails(vectors, defect):
    client, payloads, point, original = vectors
    assert list(point_digests(client, payloads)) == [POINT]
    if defect == "missing":
        client.scroll.return_value = ([], None)
    elif defect == "duplicate":
        client.scroll.return_value = ([point, point], None)
    elif defect == "unexpected":
        point.id = "00000000-0000-0000-0000-000000000002"
    elif defect == "stale":
        point.payload["workspace_id"] = True  # Bool is not an integer payload ID.
    elif defect == "short":
        point.vector.pop()
    else:
        point.vector[0] = -1.0
    with pytest.raises(SeedError):
        point_digests(client, payloads, {POINT: original})


def test_failed_collection_create_never_upserts_or_deletes(vectors):
    client, payloads, _, original = vectors
    client.create_collection.return_value = False
    embedding = Mock()
    embedding.embed_texts.return_value = [original]
    with pytest.raises(SeedError, match="did not create"):
        create_vectors(client, payloads, embedding)
    assert "upsert" not in [item[0] for item in client.method_calls]
    assert "delete_collection" not in [item[0] for item in client.method_calls]


def test_interrupted_upsert_leaves_owned_collection_without_cleanup(vectors):
    client, payloads, _, original = vectors
    client.upsert.side_effect = TimeoutError
    embedding = Mock()
    embedding.embed_texts.return_value = [original]
    with pytest.raises(TimeoutError):
        create_vectors(client, payloads, embedding)
    assert client.create_collection.call_args.kwargs["collection_name"] == COLLECTION
    assert "delete_collection" not in [item[0] for item in client.method_calls]


@pytest.mark.parametrize("collection_exists", [True, False])
def test_partial_seed_is_rejected_before_writes(monkeypatch, prepared, identity, collection_exists):
    connection = Mock()
    connection.exec_driver_sql.return_value.scalar_one.return_value = True
    client = Mock()
    client.collection_exists.return_value = collection_exists
    config = Mock(get_current_head=Mock(return_value="head"))
    monkeypatch.setattr(seeder.ScriptDirectory, "from_config", Mock(return_value=config))
    inspector = Mock()
    inspector.get_table_names.return_value = [] if collection_exists else [seeder.SEED_STATE.name]
    monkeypatch.setattr(seeder, "inspect", Mock(return_value=inspector))
    current_seed = Mock()
    monkeypatch.setattr(seeder, "current_seed", current_seed)
    with pytest.raises(SeedError, match="readiness and Qdrant collection disagree"):
        seeder.seed_transaction(connection, client, SimpleNamespace(identity=identity, verify_current_identity=Mock()), prepared, {}, Mock())
    current_seed.assert_not_called()
    client.create_collection.assert_not_called()


def test_concurrent_attempt_stops_before_collection_access(prepared, identity):
    connection = Mock()
    connection.exec_driver_sql.return_value.scalar_one.return_value = False
    client = Mock()
    targets = SimpleNamespace(identity=identity, verify_current_identity=Mock())
    with pytest.raises(SeedError, match="Another worker"):
        seeder.seed_transaction(connection, client, targets, prepared, {}, Mock())
    client.collection_exists.assert_not_called()


def test_failed_seed_rolls_back_sql_and_never_returns_readiness(
    monkeypatch, tmp_path, test_engine, prepared, identity,
):
    connection = SimpleNamespace(close=Mock())
    engine = SimpleNamespace(begin=test_engine.begin, dispose=Mock())
    targets = SimpleNamespace(engine=lambda: engine, qdrant=lambda: connection)
    monkeypatch.setattr(seeder, "prepare_corpus", lambda *args: prepared)
    monkeypatch.setattr(seeder, "verified_model", lambda *args: (Path(tmp_path), {}))

    def fail_after_sql_write(sql, *args):
        with seeder.Session(bind=sql) as session:
            seeder.insert_corpus(session, prepared)
            assert session.scalar(select(func.count()).select_from(Document)) == 8
        raise TimeoutError

    monkeypatch.setattr(seeder, "seed_transaction", fail_after_sql_write)
    with pytest.raises(SeedError, match="TimeoutError"):
        seeder.seed(targets, tmp_path)
    with test_engine.connect() as sql:
        assert sql.scalar(select(func.count()).select_from(Document)) == 0
    connection.close.assert_called_once_with()


def test_cli_failure_only_writes_diagnostic(monkeypatch, capsys):
    monkeypatch.setattr(seeder, "resolve_targets", Mock(side_effect=SeedError("target not verified")))
    assert seeder.main([
        "--docker-context", "local", "--project", PROJECT, "--postgres-port", "15432",
        "--qdrant-port", "16333", "--model-dir", "unused",
    ]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "target not verified" in output.err





def test_qdrant_transport_binds_request_to_container_and_start_token(monkeypatch, identity):
    from evaluation.portable_benchmark import seed_transport

    execute = Mock(return_value=(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n\r\nB\r\n{\"ok\":true}\r\n0\r\n\r\n"
    ))
    monkeypatch.setattr(seed_transport, "exec_qdrant", execute)
    transport = ContainerQdrantTransport("local", identity.qdrant_container, identity.qdrant_instance)
    with httpx.Client(transport=transport) as client:
        response = client.post("http://127.0.0.1:6333/collections?wait=true", json={"value": 1})
    assert response.json() == {"ok": True}
    context, container, script, instance, length = execute.call_args.args
    assert context == "local" and container == identity.qdrant_container and instance == identity.qdrant_instance
    assert script.index('[[ "$instance"') < script.index("/dev/tcp")
    raw = execute.call_args.kwargs["data"]
    assert int(length) == len(raw)
    assert raw.startswith(b"POST /collections?wait=true HTTP/1.1\r\n")
    assert b"Connection: close\r\n" in raw and raw.endswith(b'\r\n\r\n{"value":1}')


@pytest.mark.parametrize("restart_same_id", [False, True])
def test_qdrant_replacement_using_same_host_port_and_key_is_rejected(monkeypatch, identity, restart_same_id):
    from evaluation.portable_benchmark import seed_transport

    targets = Targets(identity, "p" * 64, "k" * 64)
    active_id = identity.qdrant_container
    active_instance = identity.qdrant_instance
    forwarded = []

    def execute(context, container, script, expected, length, *, data):
        if (container, expected) != (active_id, active_instance):
            raise ContainerTransportError("Qdrant instance changed")
        forwarded.append(data)
        result = b'{"result":{"collections":[]},"status":"ok","time":0}'
        return b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(result)).encode() + b"\r\n\r\n" + result

    monkeypatch.setattr(seed_transport, "exec_qdrant", execute)
    with httpx.Client(transport=targets.qdrant_transport()) as client:
        url = "http://127.0.0.1:6333/collections"
        auth = {"api-key": targets.qdrant_key}
        assert client.get(url, headers=auth).status_code == 200
        active_id = identity.qdrant_container if restart_same_id else "c" * 64
        active_instance = "pid:[18]:105"
        for method in ("PUT", "DELETE"):
            with pytest.raises(ContainerTransportError, match="changed"):
                client.request(method, url + "/aura_portable_passages_v1", headers=auth)
    assert len(forwarded) == 1  # No write or cleanup went to the replacement.


def test_transport_maps_exec_identity_failure_without_leaking_request(monkeypatch, identity):
    from evaluation.portable_benchmark import seed_transport

    execute = Mock(return_value=SimpleNamespace(returncode=47, stdout=b"", stderr=b"private stderr"))
    monkeypatch.setattr(seed_transport.subprocess, "run", execute)
    with pytest.raises(ContainerTransportError, match="identity changed") as error:
        seed_transport.exec_qdrant("local", identity.qdrant_container, "guard", "token", data=b"private stdin")
    assert "private" not in str(error.value)
    assert not any("private" in argument for argument in execute.call_args.args[0])


@pytest.mark.parametrize("which", ["postgres", "qdrant", "qdrant_restart"])
def test_final_target_check_rejects_replacement(monkeypatch, identity, which):
    changes = {
        "postgres": {"postgres_container": "c" * 64},
        "qdrant": {"qdrant_container": "d" * 64},
        "qdrant_restart": {"qdrant_instance": "pid:[19]:105"},
    }
    original = Targets(identity, "p" * 64, "q" * 64)
    next_target = Targets(ResourceIdentity.model_validate(identity.model_dump() | changes[which]), "p" * 64, "q" * 64)
    monkeypatch.setattr("evaluation.portable_benchmark.seed_infra.resolve_targets", Mock(return_value=next_target))
    connection = Mock()
    with pytest.raises(SeedError, match="identity changed"):
        original.verify_current_identity(connection)
    connection.exec_driver_sql.assert_not_called()


def test_final_identity_mismatch_prevents_writing_readiness(monkeypatch, prepared, identity):
    connection = Mock()
    connection.exec_driver_sql.return_value.scalar_one.return_value = True
    client = Mock(collection_exists=Mock(return_value=False))
    targets = SimpleNamespace(
        identity=identity,
        verify_current_identity=Mock(side_effect=[None, SeedError("Benchmark infrastructure identity changed")]),
    )
    monkeypatch.setattr(seeder.ScriptDirectory, "from_config", Mock(return_value=Mock(get_current_head=Mock(return_value="head"))))
    monkeypatch.setattr(seeder, "inspect", Mock(return_value=Mock(get_table_names=Mock(return_value=[]))))
    monkeypatch.setattr(seeder, "current_seed", Mock(return_value=None))
    monkeypatch.setattr(seeder, "insert_corpus", Mock())
    first = prepared.documents[0]
    rows = [(first.key, 1, [(first.passages[0].passage_id, 1, POINT)])]
    monkeypatch.setattr(seeder, "read_corpus", Mock(return_value=(1, 1, rows, {POINT: {"text": "example"}})))
    vectors = Mock(return_value={POINT: "d" * 64})
    monkeypatch.setattr(seeder, "create_vectors", vectors)
    readiness = Mock()
    monkeypatch.setattr(seeder.SEED_STATE, "create", readiness)
    with pytest.raises(SeedError, match="identity changed"):
        seeder.seed_transaction(
            connection, client, targets, prepared, {"repo_id": "test", "revision": "a" * 40}, Mock(),
        )
    vectors.assert_called_once()
    readiness.assert_not_called()
    connection.execute.assert_not_called()
    client.delete_collection.assert_not_called()





def test_sdk_reads_and_writes_use_the_container_transport(monkeypatch, identity):
    from contextlib import closing
    from qdrant_client.http.exceptions import ResponseHandlingException

    from evaluation.portable_benchmark import seed_transport

    record = b'{"result":{"collections":[]},"status":"ok","time":0}'
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(record)).encode() + b"\r\n\r\n" + record
    )
    execute = Mock(side_effect=[response, ContainerTransportError("instance changed")])
    monkeypatch.setattr(seed_transport, "exec_qdrant", execute)
    targets = Targets(identity, "p" * 64, "k" * 64)
    with pytest.warns(UserWarning, match="insecure connection"):
        qdrant = targets.qdrant()
    with closing(qdrant):
        assert qdrant.get_collections().collections == []
        with pytest.raises(ResponseHandlingException) as rejected:
            qdrant.create_collection(
                COLLECTION, vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
            )
    assert isinstance(rejected.value.source, ContainerTransportError)
    assert all(call.args[1] == identity.qdrant_container for call in execute.call_args_list)
    assert all(call.args[3] == identity.qdrant_instance for call in execute.call_args_list)
    assert b"api-key: " + targets.qdrant_key.encode() in execute.call_args_list[0].kwargs["data"].lower()
