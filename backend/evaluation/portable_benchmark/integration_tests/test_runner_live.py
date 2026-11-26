"""Opt-in portable ranking on two independently seeded Docker stacks."""

from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import pytest
from qdrant_client import models
from sqlalchemy import inspect

from evaluation.portable_benchmark import run_benchmark, seed, seed_lifecycle
from evaluation.portable_benchmark.corpus import CORPUS_DIR, load_corpus, passage_catalog
from evaluation.portable_benchmark.ranking_report import validate_report
from evaluation.portable_benchmark.seed_infra import COLLECTION, resolve_targets

from test_seed_live import required, unused_loopback_ports


pytestmark = pytest.mark.portable_benchmark_live


@contextmanager
def stack(path, context):
    pg_port, qdrant_port = unused_loopback_ports()
    try:
        ownership = seed_lifecycle.start(path, context, None, pg_port, qdrant_port)
        yield ownership, resolve_targets(context, ownership.project, pg_port, qdrant_port)
    finally:
        if path.is_file():
            seed_lifecycle.stop(path)


def offset_ids(target):
    engine = target.engine()
    try:
        with engine.begin() as sql:
            target.verify_postgres(sql)
            config = seed.migration_config(sql)
            revision = seed.ScriptDirectory.from_config(config).get_current_head()
            assert seed.current_seed(sql, config, revision) is None
            sql.exec_driver_sql("ALTER SEQUENCE documents_id_seq RESTART WITH 1000")
            sql.exec_driver_sql("ALTER SEQUENCE document_chunks_id_seq RESTART WITH 1000")
    finally:
        engine.dispose()


def run_cli(ownership, embedding, reranker, output):
    assert run_benchmark.main([
        "--docker-context", ownership.context, "--project", ownership.project,
        "--postgres-port", str(ownership.postgres_port), "--qdrant-port", str(ownership.qdrant_port),
        "--embedding-model-dir", str(embedding), "--reranker-model-dir", str(reranker),
        "--mode", "ranking-only", "--output", str(output),
    ]) == 0
    return validate_report(output.read_bytes(), run_benchmark.load_report_context("ranking-only"))


def logical_rankings(report):
    return {
        result.configuration.name: {
            query.query_id: [hit.passage_id for hit in query.ranking] for query in result.queries
        }
        for result in report.results
    }


def test_offline_rankings_match_independent_docker_seeds(tmp_path, monkeypatch):
    context = required("AURA_BENCHMARK_TEST_CONTEXT")
    embedding = Path(required("AURA_BENCHMARK_TEST_MODEL_DIR"))
    reranker = Path(required("AURA_BENCHMARK_TEST_RERANKER_DIR"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    manifest, sources = load_corpus(CORPUS_DIR)
    expected = {passage.passage_id for passage in passage_catalog(manifest, sources)}
    assert len(expected) == 40

    with ExitStack() as cleanup:
        first, first_target = cleanup.enter_context(stack(tmp_path / "first.json", context))
        second, second_target = cleanup.enter_context(stack(tmp_path / "second.json", context))
        offset_ids(second_target)
        records = []
        reports = []
        for i, (ownership, target) in enumerate(((first, first_target), (second, second_target))):
            record = seed.seed(target, embedding)
            records.append(record)
            assert {p.passage_id for doc in record.documents for p in doc.passages} == expected
            with closing(target.qdrant()) as client:
                assert client.count(COLLECTION, exact=True).count == 40
                assert client.create_collection(
                    "runner_canary", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
                )
                client.upsert("runner_canary", [models.PointStruct(id=1, vector=[1.0, 0.0], payload={"marker": 44})])
            engine = target.engine()
            try:
                with engine.begin() as sql:
                    target.verify_postgres(sql)
                    sql.exec_driver_sql("CREATE SCHEMA runner_canary")
                    sql.exec_driver_sql("CREATE TABLE runner_canary.marker (value integer)")
                    sql.exec_driver_sql("INSERT INTO runner_canary.marker VALUES (44)")
                report = run_cli(ownership, embedding, reranker, tmp_path / f"ranks-{i}.json")
                with engine.connect() as sql:
                    target.verify_postgres(sql)
                    assert inspect(sql).get_table_names(schema="runner_canary") == ["marker"]
                    assert sql.exec_driver_sql("SELECT value FROM runner_canary.marker").scalar_one() == 44
            finally:
                engine.dispose()
            with closing(target.qdrant()) as client:
                assert client.retrieve("runner_canary", ids=[1])[0].payload == {"marker": 44}
            assert seed.seed(target, embedding) == record
            assert report.mode == "ranking-only"
            assert report.corpus_fingerprint == manifest.corpus_fingerprint
            assert report.dataset_review_status == "draft"
            assert {result.configuration.name for result in report.results} == {
                "dense", "hybrid", "hybrid_rerank_5", "hybrid_rerank_10", "hybrid_rerank_20",
            }
            for result in report.results:
                assert len(result.queries) == 36 and result.quality is None
                assert all(
                    len(query.ranking) == 5 and {hit.passage_id for hit in query.ranking} <= expected
                    for query in result.queries
                )
            reports.append(report)

        left, right = ({p.passage_id: p for doc in record.documents for p in doc.passages}
                       for record in records)
        assert left.keys() == right.keys()
        assert all(left[key].chunk_id != right[key].chunk_id for key in expected)
        assert all(left[key].vector_id != right[key].vector_id for key in expected)
        assert logical_rankings(reports[0]) == logical_rankings(reports[1])
