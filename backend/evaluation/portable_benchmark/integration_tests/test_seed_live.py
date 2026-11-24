"""Opt-in tests for two explicitly provisioned, disposable Compose stacks."""

from contextlib import closing
import os
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import models
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from evaluation.portable_benchmark import seed as seeder
from evaluation.portable_benchmark.corpus import CORPUS_DIR
from evaluation.portable_benchmark.seed_assets import prepare_corpus
from evaluation.portable_benchmark.seed_infra import COLLECTION, resolve_targets
from evaluation.portable_benchmark.seed_vectors import DIMENSIONS, point_digests


pytestmark = pytest.mark.portable_benchmark_live


def required(name):
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"Set {name} for the opt-in portable benchmark live tests")
    return value


def target(suffix=""):
    return resolve_targets(
        required("AURA_BENCHMARK_TEST_CONTEXT"),
        required(f"AURA_BENCHMARK_TEST_PROJECT{suffix}"),
        int(required(f"AURA_BENCHMARK_TEST_PG_PORT{suffix}")),
        int(required(f"AURA_BENCHMARK_TEST_QDRANT_PORT{suffix}")),
    )


@pytest.fixture(scope="module")
def model_dir():
    return Path(required("AURA_BENCHMARK_TEST_MODEL_DIR"))


@pytest.fixture(scope="module")
def first_target():
    return target()


@pytest.fixture(scope="module")
def seeded(first_target, model_dir):
    engine = first_target.engine()
    # Canary resources live only in the already verified disposable containers.
    schema = "canary_" + uuid4().hex
    collection = "canary_" + uuid4().hex
    with closing(first_target.qdrant()) as qdrant:
        created_schema = False
        created_collection = False
        try:
            with engine.begin() as sql:
                first_target.verify_connections(sql)
                assert not inspect(sql).get_table_names(schema="public")
                assert not qdrant.collection_exists(COLLECTION)
                sql.exec_driver_sql(f"CREATE SCHEMA {schema}")
                sql.exec_driver_sql(f"CREATE TABLE {schema}.marker (value integer)")
                sql.exec_driver_sql(f"INSERT INTO {schema}.marker VALUES (47)")
            created_schema = True
            assert qdrant.create_collection(
                collection, vectors_config=models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE),
            )
            created_collection = True
            point = models.PointStruct(id=1, vector=[1.0] + [0.0] * (DIMENSIONS - 1), payload={"marker": 47})
            qdrant.upsert(collection, [point], wait=True)
            record = seeder.seed(first_target, model_dir)
            yield record
            with engine.connect() as sql:
                assert sql.exec_driver_sql(f"SELECT value FROM {schema}.marker").scalar_one() == 47
            assert qdrant.retrieve(collection, [1])[0].payload == {"marker": 47}
        finally:
            try:
                if created_collection:
                    qdrant.delete_collection(collection)
            finally:
                try:
                    if created_schema:
                        with engine.begin() as sql:
                            first_target.verify_postgres(sql)
                            sql.exec_driver_sql(f"DROP TABLE {schema}.marker")
                            sql.exec_driver_sql(f"DROP SCHEMA {schema}")
                finally:
                    engine.dispose()


def test_fresh_seed_stores_every_passage_and_preserves_canaries(seeded):
    assert len(seeded.documents) == 8
    assert sum(len(document.passages) for document in seeded.documents) == 40
    assert all(item.vector_id for document in seeded.documents for item in document.passages)


def test_full_sql_and_qdrant_consistency(first_target, seeded, tmp_path):
    expected = prepare_corpus(CORPUS_DIR, tmp_path / "pdfs")
    engine = first_target.engine()
    try:
        with Session(engine) as sql, closing(first_target.qdrant()) as qdrant:
            _, _, rows, payloads = seeder.read_corpus(sql, expected, seeded.resources)
            digests = point_digests(qdrant, payloads)
        assert len(rows) == 8 and len(digests) == 40
        assert digests == {
            item.vector_id: item.vector_sha256
            for document in seeded.documents for item in document.passages
        }
    finally:
        engine.dispose()


def test_second_run_reuses_identical_readiness(first_target, model_dir, seeded):
    assert seeder.seed(first_target, model_dir) == seeded


def test_independent_seed_has_same_logical_ids_and_different_runtime_ids(seeded, model_dir):
    second = target("_SECOND")
    assert second.identity.postgres_container != seeded.resources.postgres_container
    assert second.identity.qdrant_container != seeded.resources.qdrant_container
    engine = second.engine()
    try:
        with engine.begin() as sql, closing(second.qdrant()) as qdrant:
            second.verify_connections(sql)
            assert not inspect(sql).get_table_names(schema="public")
            assert not qdrant.collection_exists(COLLECTION)
            config = seeder.migration_config(sql)
            revision = seeder.ScriptDirectory.from_config(config).get_current_head()
            assert seeder.current_seed(sql, config, revision) is None
            sql.exec_driver_sql("ALTER SEQUENCE documents_id_seq RESTART WITH 1000")
            sql.exec_driver_sql("ALTER SEQUENCE document_chunks_id_seq RESTART WITH 1000")
    finally:
        engine.dispose()
    alternative = seeder.seed(second, model_dir)
    assert alternative.corpus_fingerprint == seeded.corpus_fingerprint
    for primary, other in zip(seeded.documents, alternative.documents, strict=True):
        assert primary.document_key == other.document_key
        assert primary.document_id != other.document_id
        for a, b in zip(primary.passages, other.passages, strict=True):
            assert a.passage_id == b.passage_id
            assert a.chunk_id != b.chunk_id
            assert a.vector_id != b.vector_id





def unused_loopback_ports():
    import socket

    with socket.socket() as pg, socket.socket() as qdrant:
        for listener in (pg, qdrant):
            listener.bind(("127.0.0.1", 0))
        return pg.getsockname()[1], qdrant.getsockname()[1]


def test_lifecycle_refuses_a_real_same_project_canary(tmp_path):
    from evaluation.portable_benchmark import seed_lifecycle as lifecycle
    from evaluation.portable_benchmark.seed_infra import QDRANT_IMAGE, SeedError, docker, local_context

    context = required("AURA_BENCHMARK_TEST_CONTEXT")
    local_context(context)
    execution = uuid4().hex
    project = f"aura-benchmark-{execution}"
    label = f"com.docker.compose.project={project}"
    canary = f"{project}-canary"
    target_file = tmp_path / "collision.json"
    container = network = volume = None
    try:
        container = docker(
            context, "container", "create", "--name", canary, "--entrypoint", "/bin/true",
            "--network", "none", "--tmpfs", "/qdrant/storage", "--label", label,
            "--label", "com.docker.compose.service=postgres",
            "--label", "com.docker.compose.oneoff=False",
            "--label", "com.docker.compose.config-hash=unrelated-canary",
            QDRANT_IMAGE, timeout=300,
        ).strip()
        network = docker(context, "network", "create", "--label", label, f"{canary}-net").strip()
        volume = docker(
            context, "volume", "create", "--label", label,
            "--label", f"org.aura.test-canary={execution}", f"{canary}-vol",
        ).strip()
        with pytest.raises(SeedError, match="Existing resources"):
            lifecycle.start(target_file, context, project, *unused_loopback_ports())
        with pytest.raises(SeedError, match="ownership record"):
            lifecycle.stop(target_file)
        assert docker(context, "container", "inspect", "--format", "{{.Id}}", container).strip() == container
        assert docker(context, "network", "inspect", "--format", "{{.Id}}", network).strip() == network
        assert docker(context, "volume", "inspect", "--format", "{{.Name}}", volume).strip() == volume
    finally:
        if container:
            docker(context, "container", "rm", container)
        if network:
            docker(context, "network", "rm", network)
        if volume:
            import json

            data, = json.loads(docker(context, "volume", "inspect", volume))
            assert data["Labels"]["org.aura.test-canary"] == execution
            docker(context, "volume", "rm", volume)


def test_actual_replacement_on_same_port_and_key_is_never_seeded(
    tmp_path, monkeypatch, model_dir,
):
    import httpx
    from time import monotonic, sleep

    from evaluation.portable_benchmark import seed_lifecycle as lifecycle
    from evaluation.portable_benchmark.seed_infra import QDRANT_IMAGE, SeedError, docker, resolve_targets

    path = tmp_path / "replacement.json"
    context = required("AURA_BENCHMARK_TEST_CONTEXT")
    pg_port, qd_port = unused_loopback_ports()
    replacement = None
    state = None
    try:
        state = lifecycle.start(path, context, None, pg_port, qd_port)
        targets = resolve_targets(context, state.project, pg_port, qd_port)
        original = state.containers["qdrant"]
        original_model = seeder.LocalEmbeddingService
        url = f"http://127.0.0.1:{qd_port}/collections"
        headers = {"api-key": targets.qdrant_key}

        def replace_after_checks(directory):
            nonlocal replacement
            docker(context, "container", "stop", "--time", "10", original)
            docker(context, "container", "rm", original)
            replacement = docker(
                context, "container", "run", "-d", "--name", f"{state.project}-replacement",
                "--tmpfs", "/qdrant/storage", "--publish", f"127.0.0.1:{qd_port}:6333",
                "-e", "QDRANT__SERVICE__API_KEY", QDRANT_IMAGE,
                env=dict(os.environ, QDRANT__SERVICE__API_KEY=targets.qdrant_key), timeout=60,
            ).strip()
            deadline = monotonic() + 20
            with httpx.Client(trust_env=False, timeout=2) as client:
                while monotonic() < deadline:
                    try:
                        if client.get(url, headers=headers).status_code == 200:
                            return original_model(directory)
                    except httpx.TransportError:
                        pass
                    sleep(0.2)
            pytest.fail("Replacement did not start at the original endpoint with the original key")

        monkeypatch.setattr(seeder, "LocalEmbeddingService", replace_after_checks)
        with pytest.raises(SeedError, match="inspected Qdrant container"):
            seeder.seed(targets, model_dir)
        assert replacement and replacement != original

        engine = targets.engine()
        try:
            with engine.connect() as connection:
                targets.verify_postgres(connection)
                assert connection.exec_driver_sql("SELECT to_regclass('public.aura_portable_seed')").scalar_one() is None
        finally:
            engine.dispose()
        with httpx.Client(trust_env=False, timeout=5) as client:
            response = client.get(url, headers=headers)
            assert response.status_code == 200 and response.json()["result"]["collections"] == []
        with pytest.raises(SeedError, match="unowned|replaced"):
            lifecycle.stop(path)
        assert docker(context, "container", "inspect", "--format", "{{.Id}}", replacement).strip() == replacement
    finally:
        if replacement:
            docker(context, "container", "stop", "--time", "10", replacement)
            docker(context, "container", "rm", replacement)
        if path.is_file():
            lifecycle.stop(path)
