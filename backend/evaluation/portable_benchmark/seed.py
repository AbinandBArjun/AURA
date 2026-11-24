"""Seed only the explicitly verified disposable PostgreSQL and Qdrant pair."""

import argparse
from contextlib import closing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import CheckConstraint, Column, Integer, MetaData, Table, func, inspect, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models import Document, DocumentChunk, DocumentPage, User, Workspace
from evaluation.portable_benchmark.corpus import CORPUS_DIR, json_digest
from evaluation.portable_benchmark.seed_assets import LocalEmbeddingService, prepare_corpus, verified_model
from evaluation.portable_benchmark.seed_infra import COLLECTION, SeedError, resolve_targets
from evaluation.portable_benchmark.seed_record import DocumentMapping, PassageMapping, SeedRecord
from evaluation.portable_benchmark.seed_transport import ContainerTransportError
from evaluation.portable_benchmark.seed_vectors import DIMENSIONS, create_vectors, point_digests


OWNER_EMAIL = "portable-benchmark@example.invalid"
OWNER_NAME = "AURA Portable Benchmark"
WORKSPACE_NAME = "AURA Portable Retrieval Benchmark"
LOCK_ID = 7304201743
SEED_STATE = Table(
    "aura_portable_seed", MetaData(), Column("id", Integer, primary_key=True),
    Column("record", JSONB, nullable=False), CheckConstraint("id = 1"),
)


def migration_config(connection):
    config = Config()
    config.set_main_option("script_location", str(CORPUS_DIR / "seed_migrations").replace("%", "%%"))
    versions = CORPUS_DIR.parents[1] / "migrations" / "versions"
    config.set_main_option("version_locations", str(versions).replace("%", "%%"))
    config.set_main_option("path_separator", "os")
    config.attributes["connection"] = connection
    return config


def current_seed(connection, config, revision):
    found = inspect(connection)
    tables = set(found.get_table_names(schema="public"))
    expected = set(Base.metadata.tables) | {"alembic_version"}
    if found.get_view_names(schema="public") or found.get_materialized_view_names(schema="public"):
        raise SeedError("Unexpected views in the benchmark database. Use a fresh benchmark project")
    if not tables:
        command.upgrade(config, revision)
        return None
    if tables not in (expected, expected | {SEED_STATE.name}):
        raise SeedError("Unexpected or partial benchmark database schema. Use a fresh benchmark project")
    if MigrationContext.configure(connection).get_current_heads() != (revision,):
        raise SeedError("The benchmark database migration revision differs from the checked-in head")
    if SEED_STATE.name in tables:
        rows = connection.execute(select(SEED_STATE.c.record)).scalars().all()
        if len(rows) != 1:
            raise SeedError("The benchmark readiness table is incomplete")
        try:
            return SeedRecord.model_validate(rows[0])
        except ValueError as error:
            raise SeedError("Invalid stored benchmark readiness record") from error
    for table in Base.metadata.sorted_tables:
        if connection.scalar(select(func.count()).select_from(table)):
            raise SeedError("Benchmark rows exist without readiness. Use a fresh benchmark project")
    return None


def insert_corpus(session, corpus):
    owner = User(email=OWNER_EMAIL, display_name=OWNER_NAME)
    workspace = Workspace(name=WORKSPACE_NAME, owner=owner)
    for document in corpus.documents:
        row = Document(
            filename=f"{document.key}.pdf", workspace=workspace,
            content_type="application/pdf", status="processed",
            size_bytes=document.pdf_bytes, checksum_sha256=document.pdf_sha256,
            storage_key=f"portable-benchmark/{document.key}.pdf",
        )
        row.pages = [
            DocumentPage(page_number=page["page_number"], text=page["text"], char_count=len(page["text"]))
            for page in document.pages
        ]
        row.chunks = [DocumentChunk(**vars(passage.chunk)) for passage in document.passages]
    session.add(owner)
    session.flush()


def read_corpus(session, corpus, identity):
    owners = session.scalars(select(User)).all()
    workspaces = session.scalars(select(Workspace)).all()
    if (
        len(owners) != 1 or owners[0].email != OWNER_EMAIL or owners[0].display_name != OWNER_NAME
        or len(workspaces) != 1 or workspaces[0].name != WORKSPACE_NAME or workspaces[0].owner_id != owners[0].id
    ):
        raise SeedError("Benchmark owner or workspace is missing or unexpected")
    owner, workspace = owners[0], workspaces[0]
    rows = session.scalars(select(Document).order_by(Document.filename)).all()
    if len(rows) != len(corpus.documents):
        raise SeedError("Unexpected number of benchmark documents")
    point_payloads = {}
    document_rows = []
    for document, row in zip(corpus.documents, rows):
        filename = f"{document.key}.pdf"
        if (
            row.workspace_id != workspace.id or row.filename != filename or row.content_type != "application/pdf"
            or row.status != "processed" or row.extraction_error is not None
            or row.size_bytes != document.pdf_bytes or row.checksum_sha256 != document.pdf_sha256
            or row.storage_key != f"portable-benchmark/{filename}"
        ):
            raise SeedError(f"Benchmark document metadata differs for {document.key}")
        pages = sorted(row.pages, key=lambda item: item.page_number)
        if [(page.page_number, page.text, page.char_count) for page in pages] != [
            (page["page_number"], page["text"], len(page["text"])) for page in document.pages
        ]:
            raise SeedError(f"Benchmark page text or metadata differs for {document.key}")
        chunks = sorted(row.chunks, key=lambda item: item.chunk_index)
        if len(chunks) != len(document.passages):
            raise SeedError(f"Unexpected number of chunks for {document.key}")
        chunk_rows = []
        for passage, chunk in zip(document.passages, chunks):
            attributes = vars(passage.chunk)
            if any(getattr(chunk, name) != value for name, value in attributes.items()):
                raise SeedError(f"Benchmark SQL chunk differs for {passage.passage_id}")
            point = None
            if chunk.retrievable:
                point = str(uuid5(NAMESPACE_URL, f"aura-portable:{identity.postgres_system_id}:{passage.passage_id}"))
                point_payloads[point] = {
                    "passage_id": passage.passage_id, "document_key": document.key,
                    "workspace_id": workspace.id, "document_id": row.id, "chunk_id": chunk.id,
                    **attributes,
                }
            chunk_rows.append((passage.passage_id, chunk.id, point))
        document_rows.append((document.key, row.id, chunk_rows))
    return owner.id, workspace.id, document_rows, point_payloads


def seed_transaction(connection, client, targets, corpus, model_manifest, make_embedding):
    targets.verify_current_identity(connection)
    connection.exec_driver_sql("SET LOCAL search_path TO public")
    connection.exec_driver_sql("SET LOCAL statement_timeout TO '30s'")
    if not connection.exec_driver_sql(f"SELECT pg_try_advisory_xact_lock({LOCK_ID})").scalar_one():
        raise SeedError("Another worker is seeding or verifying this benchmark database")
    config = migration_config(connection)
    revision = ScriptDirectory.from_config(config).get_current_head()

    # Refuse orphaned collections before applying migrations or creating rows.
    existing = client.collection_exists(COLLECTION)
    tables = set(inspect(connection).get_table_names(schema="public"))
    if existing != (SEED_STATE.name in tables):
        raise SeedError("PostgreSQL readiness and Qdrant collection disagree. Use a fresh benchmark project")
    saved = current_seed(connection, config, revision)
    with Session(bind=connection) as session:
        if saved is None:
            insert_corpus(session, corpus)
        elif saved.resources != targets.identity:
            raise SeedError("Stored readiness refers to different benchmark resources")

        owner_id, workspace_id, rows, payloads = read_corpus(session, corpus, targets.identity)
        if saved is None:
            digests = create_vectors(client, payloads, make_embedding())
        else:
            digests = point_digests(client, payloads)
        fresh = SeedRecord(
            schema_version=1, corpus_fingerprint=corpus.fingerprint, corpus_version=corpus.corpus_version,
            extraction_protocol=corpus.extraction, chunking_protocol=corpus.chunking,
            protocol_fingerprint=corpus.protocol, passage_id_scheme="p1", alembic_revision=revision,
            model_id=model_manifest["repo_id"], model_revision=model_manifest["revision"],
            model_manifest_sha256=json_digest(model_manifest), vector_dimensions=DIMENSIONS,
            resources=targets.identity, owner_id=owner_id, workspace_id=workspace_id,
            documents=[
                DocumentMapping(
                    document_key=key, document_id=document_id,
                    passages=[
                        PassageMapping(
                            passage_id=pid, chunk_id=cid, vector_id=point,
                            vector_sha256=digests[point] if point is not None else None,
                        )
                        for pid, cid, point in chunks
                    ],
                )
                for key, document_id, chunks in rows
            ],
        )
        targets.verify_current_identity(connection)
        if saved is None:
            SEED_STATE.create(connection)
            connection.execute(SEED_STATE.insert().values(id=1, record=fresh.model_dump()))
        elif fresh != saved:
            raise SeedError("Stored readiness disagrees with the verified corpus, runtime mappings, or vectors")
        return fresh


def seed(targets, model_directory, corpus_dir=CORPUS_DIR):
    with TemporaryDirectory(prefix="aura-portable-seed-") as directory:
        corpus = prepare_corpus(corpus_dir, Path(directory) / "pdfs")
    model_path, manifest = verified_model(model_directory)
    engine = targets.engine()
    try:
        with closing(targets.qdrant()) as client, engine.begin() as connection:
            record = seed_transaction(connection, client, targets, corpus, manifest, lambda: LocalEmbeddingService(model_path))
        return record
    except SeedError:
        raise
    except Exception as error:
        cause = error.source if isinstance(error, ResponseHandlingException) else error
        if isinstance(cause, ContainerTransportError):
            raise SeedError(str(cause)) from error
        raise SeedError(
            f"Seeding failed during SQL or vector verification ({type(error).__name__}). "
            "Check benchmark service health. A partial collection requires a fresh benchmark project."
        ) from error
    finally:
        engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--postgres-port", required=True, type=int)
    parser.add_argument("--qdrant-port", required=True, type=int)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        targets = resolve_targets(args.docker_context, args.project, args.postgres_port, args.qdrant_port)
        record = seed(targets, args.model_dir)
    except SeedError as error:
        print(f"Portable benchmark seed failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Portable benchmark asset validation failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Portable benchmark preparation failed ({type(error).__name__}). Check local assets and service readiness.", file=sys.stderr)
        return 1
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
