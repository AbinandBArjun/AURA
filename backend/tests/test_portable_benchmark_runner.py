from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evaluation.portable_benchmark import ranking, ranking_report, report_output, reranker_assets, run_benchmark, seed
from evaluation.portable_benchmark.corpus import CORPUS_DIR, load_corpus, passage_catalog, protocol_fingerprint, read_json
from evaluation.portable_benchmark.ranking_report import BenchmarkReport, configurations, evaluate
from evaluation.portable_benchmark.schema import Judgment, PortableDataset
from evaluation.portable_benchmark.seed_assets import prepare_corpus
from evaluation.portable_benchmark.seed_infra import SeedError


MANIFEST, _ = load_corpus(CORPUS_DIR)
PROTOCOL = protocol_fingerprint(MANIFEST)
CORPUS = MANIFEST.corpus_fingerprint
CONFIGS = ["dense", "hybrid", "hybrid_rerank_5", "hybrid_rerank_10", "hybrid_rerank_20"]


def logical(index):
    return f"p1:example:{index}:{PROTOCOL}:{index:064x}"


def judgment(index, grade, **changes):
    return Judgment.model_validate({
        "passage_id": logical(index), "grade": grade, "method": "human",
        "assessor_id": "assessor", "review_status": "reviewed", "reviewer_id": "reviewer",
    } | changes)


@pytest.fixture(scope="module")
def bundle():
    manifest, sources = load_corpus(CORPUS_DIR)
    return manifest, passage_catalog(manifest, sources)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return prepare_corpus(CORPUS_DIR, tmp_path_factory.mktemp("ranking-corpus") / "pdfs")


@pytest.mark.parametrize(
    ("top_k", "hybrid", "pools"),
    [(0, None, (5, 10, 20)), (True, None, (5, 10, 20)), (41, None, (5, 10, 20)),
     (6, None, (5, 10, 20)), (5, 4, (5, 10, 20)), (5, None, (5, 5, 20)),
     (5, None, (20, 5, 10)), (5, None, (5, 10, 41)), (5, None, (5, 10)),
     (1, 40.0, (5, 10, 20))],
)
def test_invalid_profile_inputs(top_k, hybrid, pools):
    with pytest.raises(ValueError):
        configurations(top_k, 40, hybrid, pools)


def test_default_and_custom_profiles():
    assert [profile.name for profile in configurations(5, 40)] == CONFIGS
    assert [profile.candidate_pool_size for profile in configurations(6, 40, 13, (6, 9, 21))] == [None, 13, 6, 9, 21]


@pytest.mark.parametrize("defect", ["fingerprint", "corpus", "query", "pending", "missing-sha"])
def test_dataset_pinning_and_review(bundle, tmp_path, defect):
    manifest, catalog = bundle
    original = read_json(CORPUS_DIR / "dataset.json")
    external = deepcopy(original)
    if defect == "corpus":
        external["corpus_fingerprint"] = "a" * 64
    elif defect == "query":
        external["queries"].pop()
    elif defect == "pending":
        external["review_status"] = "reviewed"
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(external))
    expected = run_benchmark.dataset_fingerprint(PortableDataset.model_validate(original))
    if defect == "query":
        expected = run_benchmark.dataset_fingerprint(PortableDataset.model_validate(external))
    if defect == "fingerprint":
        expected = "0" * 64
    elif defect == "missing-sha":
        expected = None
    with pytest.raises(ValueError):
        run_benchmark.select_dataset(manifest, catalog, "ranking-only", path, expected)


def test_draft_is_only_ranking_only(bundle):
    manifest, catalog = bundle
    dataset = run_benchmark.select_dataset(manifest, catalog, "ranking-only")
    assert len(dataset.queries) == 36 and not any(query.judgments for query in dataset.queries)
    with pytest.raises(ValueError, match="quality for reviewed"):
        run_benchmark.select_dataset(manifest, catalog, "quality")


def test_default_dataset_fingerprint_is_required(monkeypatch, bundle):
    monkeypatch.setattr(run_benchmark, "DRAFT_DATASET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_benchmark.select_dataset(*bundle, "ranking-only")


def test_binary_and_graded_metrics_with_unjudged_results():
    metrics = ranking_report.quality_scores(
        [logical(3), logical(1), logical(2), logical(0)], [judgment(0, 3), judgment(1, 1), judgment(2, 0)], 3,
    )
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == pytest.approx(0.5)
    assert metrics.mrr_at_k == pytest.approx(0.5)
    from math import log2
    assert metrics.ndcg_at_k == pytest.approx((1 / log2(3)) / (7 + 1 / log2(3)))


@pytest.mark.parametrize("judgments", [
    [], [judgment(1, 0)], [judgment(1, 1, review_status="pending", reviewer_id=None)],
    [judgment(1, 1), judgment(1, 2)],
])
def test_quality_refuses_incomplete_or_duplicate_judgments(judgments):
    with pytest.raises(ValueError, match="requires unique reviewed"):
        ranking_report.quality_scores([logical(1)], judgments, 5)


def readiness(corpus, *, reverse=False, offset=0, workspace_id=7):
    documents = []
    for position, document in enumerate(corpus.documents, 1):
        passages = []
        for index, passage in enumerate(document.passages, 1):
            number = offset + position * 50 + index
            passages.append(SimpleNamespace(
                passage_id=passage.passage_id, chunk_id=number,
                vector_id=str(number) if passage.chunk.retrievable else None,
            ))
        documents.append(SimpleNamespace(document_key=document.key, document_id=offset + position, passages=passages))
    if reverse:
        documents.reverse()
    return SimpleNamespace(workspace_id=workspace_id, documents=documents)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "missing-vector", "duplicate-vector", "wrong-document"])
def test_logical_mapping_rejects_inconsistent_readiness(prepared, defect):
    record = readiness(prepared)
    source, target = record.documents[:2]
    if defect == "missing":
        source.passages.pop()
    elif defect == "duplicate":
        target.passages[0].chunk_id = source.passages[0].chunk_id
    elif defect == "unknown":
        target.passages[0].passage_id = logical(0)
    elif defect == "missing-vector":
        target.passages[0].vector_id = None
    elif defect == "duplicate-vector":
        target.passages[0].vector_id = source.passages[0].vector_id
    else:
        target.document_key = "unknown"
    with pytest.raises(SeedError, match="[Mm]apping"):
        ranking.LogicalMapping(record, prepared.documents)


def test_dense_and_rerank_ties_precede_candidate_cutoff_across_runtime_ids(prepared):
    winners = []
    embedding = Mock(embed_text=Mock(return_value=[1.0] + [0.0] * 383))
    for offset, reverse in ((0, False), (2000, True)):
        mapping = ranking.LogicalMapping(readiness(prepared, offset=offset, reverse=reverse), prepared.documents)
        points = [SimpleNamespace(id=key, score=0.5, payload=value) for key, value in reversed(list(mapping.points.items()))]
        client = Mock(query_points=Mock(return_value=SimpleNamespace(points=points)))
        ranked = ranking.DenseAdapter(client, embedding, mapping).search("query", 7, 5)
        assert client.query_points.call_args.kwargs["limit"] == 40
        assert client.query_points.call_args.kwargs["search_params"].exact
        winners.append([item["chunk_id"] for item in ranked])
        model = Mock(rerank=Mock(side_effect=lambda query, rows, limit: [
            row | {"rerank_score": 0.25} for row in reversed(rows)
        ]))
        picked = ranking.RerankAdapter(model).rerank("query", ranked[::-1], 2)
        assert [row["chunk_id"] for row in picked] == winners[-1][:2]
        assert model.rerank.call_args.kwargs["limit"] == 5
    assert winners[0] == winners[1]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "stale", "nan"])
def test_dense_validates_entire_result_set(prepared, defect):
    mapping = ranking.LogicalMapping(readiness(prepared), prepared.documents)
    points = [SimpleNamespace(id=key, score=0.5, payload=value.copy()) for key, value in mapping.points.items()]
    if defect == "missing":
        points.pop()
    elif defect == "duplicate":
        points.append(points[0])
    elif defect == "unknown":
        points[0].id = "none"
    elif defect == "stale":
        points[0].payload["chunk_id"] = True
    else:
        points[0].score = float("nan")
    with pytest.raises(SeedError):
        mapping.dense(points)


def test_sparse_ties_use_logical_order_before_cutoff(prepared):
    mapping = ranking.LogicalMapping(readiness(prepared), prepared.documents)
    adapter = ranking.SparseAdapter(mapping)
    adapter.retriever = Mock(search=Mock(return_value=[
        {**value, "score": 1.0} for value in reversed(list(mapping.chunks.values()))
    ]))
    ranked = adapter.search(None, mapping.workspace_id, "cache", 5)
    expected = sorted(value["passage_id"] for value in mapping.points.values())
    assert [row["chunk_id"] for row in ranked] == expected[:5]
    assert adapter.retriever.search.call_args.kwargs["limit"] == 40


def dataset_for_report(reviewed=False):
    return PortableDataset.model_validate({
        "schema_version": 1, "corpus_fingerprint": CORPUS,
        "review_status": "reviewed" if reviewed else "draft",
        "queries": [
            {"query_id": "one", "category": "economics", "query": "text", "judgments":
                [judgment(0, 3).model_dump()] if reviewed else []},
            {"query_id": "two", "category": "software-engineering", "query": "text", "judgments":
                [judgment(9, 1).model_dump()] if reviewed else []},
        ],
    })


def report_context(reviewed=False):
    dataset = dataset_for_report(reviewed)
    return ranking_report.ReportContext(
        MANIFEST, frozenset(logical(index) for index in range(40)), dataset,
        "quality" if reviewed else "ranking-only", 5, tuple(configurations(5, 40)),
        ranking_report.ModelIdentity(
            repo_id="sentence-transformers/all-MiniLM-L6-v2", revision="a" * 40, manifest_sha256=CORPUS,
        ),
        ranking_report.ModelIdentity(
            repo_id=reranker_assets.MODEL_ID, revision=reranker_assets.MODEL_REVISION, manifest_sha256=CORPUS,
        ),
    )


def make_report(reviewed=False):
    mode = "quality" if reviewed else "ranking-only"
    dataset = dataset_for_report(reviewed)
    profiles = configurations(5, 40)
    search = Mock(return_value=[{"chunk_id": logical(index), "score": 1.0} for index in range(5)])
    mapping = SimpleNamespace(points={index: {"passage_id": logical(index)} for index in range(40)})
    results = evaluate(SimpleNamespace(search=search), object(), 7, mapping, dataset, mode, 5, profiles)
    assert len(search.call_args_list) == 10
    context = report_context(reviewed)
    return BenchmarkReport.model_validate(dict(
        schema_version=1, evaluation_protocol="aura-portable-retrieval-v1", passage_id_scheme="p1",
        passage_protocol_sha256=PROTOCOL, corpus_name=MANIFEST.corpus_name, corpus_version=MANIFEST.corpus_version,
        corpus_fingerprint=CORPUS, dataset_fingerprint=run_benchmark.dataset_fingerprint(dataset),
        dataset_review_status=dataset.review_status, mode=mode, top_k=5, retrievable_passage_count=40,
        embedding=context.embedding, reranker=context.reranker,
        runtime=ranking_report.runtime(), limitations=["Synthetic controls"],
        results=[result.model_dump(exclude_none=True) for result in results],
    ), context=context)


def test_ranking_only_report_omits_quality_fields_and_runtime_ids():
    report = json.loads(make_report().model_dump_json(exclude_none=True))
    assert report["mode"] == "ranking-only"
    assert [entry["configuration"]["name"] for entry in report["results"]] == CONFIGS
    for entry in report["results"]:
        assert "quality" not in entry and "category_quality" not in entry
        assert set(entry["latency"]) == {"mean_ms", "median_ms", "p95_ms"}
        for query in entry["queries"]:
            assert "quality" not in query
            assert all(set(item) == {"passage_id", "score"} for item in query["ranking"])


def test_reviewed_report_macro_aggregation():
    result = make_report(reviewed=True).results[0]
    assert result.quality.precision_at_k == pytest.approx(0.1)
    assert result.quality.recall_at_k == pytest.approx(0.5)
    assert result.category_quality["economics"].mrr_at_k == 1.0
    assert result.category_quality["software-engineering"].ndcg_at_k == 0.0


def ai_assisted_dataset_payload(*, reviewed=True):
    payload = dataset_for_report(reviewed=True).model_dump()
    payload["schema_version"] = 2
    payload["review_status"] = "reviewed" if reviewed else "draft"
    record = payload["queries"][0]["judgments"][0]
    record.update(
        method="ai-assisted",
        assessor_id="ai-assessor-fixture",
        ai_provenance={"provider": "Synthetic provider", "tool": "Synthetic tool", "model": "Synthetic model"},
    )
    if reviewed:
        record["human_review"] = {
            "reviewer_kind": "human", "decision": "approved",
            "approved_on": "2026-01-01", "approval_reference": "synthetic-approval-fixture",
        }
    else:
        record.update(review_status="pending", reviewer_id=None)
    return payload


def test_v1_human_provenance_retains_canonical_payload_and_report_validation():
    dataset = dataset_for_report(reviewed=True)
    record = {
        "passage_id": logical(0), "grade": 3, "method": "human", "assessor_id": "assessor",
        "review_status": "reviewed", "reviewer_id": "reviewer",
    }
    expected = {
        "schema_version": 1, "corpus_fingerprint": CORPUS, "review_status": "reviewed",
        "queries": [
            {"query_id": "one", "category": "economics", "query": "text", "judgments": [record]},
            {"query_id": "two", "category": "software-engineering", "query": "text",
             "judgments": [record | {"passage_id": logical(9), "grade": 1}]},
        ],
    }
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert dataset.model_dump() == expected
    assert run_benchmark.dataset_fingerprint(dataset) == sha256(canonical.encode("ascii")).hexdigest()
    report = make_report(reviewed=True)
    assert ranking_report.validate_report(report.model_dump_json(), report_context(reviewed=True)) == report


def test_ai_assisted_provenance_preserves_report_protocol_and_requires_matching_fingerprint():
    dataset = PortableDataset.model_validate(ai_assisted_dataset_payload())
    original_context = report_context(reviewed=True)
    context = replace(original_context, dataset=dataset)
    report = make_report(reviewed=True)
    payload = report.model_dump()
    payload["dataset_fingerprint"] = run_benchmark.dataset_fingerprint(dataset)
    data = json.dumps(payload)
    validated = ranking_report.validate_report(data, context)
    assert validated.schema_version == report.schema_version == 1
    assert validated.evaluation_protocol == report.evaluation_protocol == "aura-portable-retrieval-v1"
    assert validated.results == report.results
    assert dataset.queries[0].judgments[0].ai_provenance.model_revision is None
    with pytest.raises(ValueError, match="dataset does not match trusted inputs"):
        ranking_report.validate_report(data, original_context)


@pytest.mark.parametrize(("field", "value", "diagnostic"), [
    ("reviewer_id", None, "different reviewer"),
    ("reviewer_id", "ai-assessor-fixture", "different reviewer"),
    ("human_review", None, "requires human approval"),
    ("human_review", {}, "Field required"),
    ("ai_provenance", None, "requires AI assessor provenance"),
    ("ai_provenance", {"provider": "Synthetic provider"}, "Field required"),
    ("ai_provenance", {"provider": "Synthetic", "tool": "Synthetic", "model_revision": "test"}, "model name"),
    ("method", "ai", "ai-assisted"),
    ("method", "human", "require the ai-assisted method"),
])
def test_ai_assisted_judgment_requires_declared_provenance_and_review(field, value, diagnostic):
    payload = ai_assisted_dataset_payload()
    payload["queries"][0]["judgments"][0][field] = value
    with pytest.raises(ValueError, match=diagnostic):
        PortableDataset.model_validate(payload)


@pytest.mark.parametrize(("field", "value"), [
    ("reviewer_kind", "ai"), ("decision", "pending"),
    ("approved_on", "2026-02-30"), ("approval_reference", " "),
])
def test_ai_assisted_approval_requires_valid_human_declarations(field, value):
    payload = ai_assisted_dataset_payload()
    payload["queries"][0]["judgments"][0]["human_review"][field] = value
    with pytest.raises(ValueError, match=field):
        PortableDataset.model_validate(payload)


def test_ai_assisted_draft_rejects_quality_mode_and_cannot_claim_reviewed_status():
    payload = ai_assisted_dataset_payload(reviewed=False)
    dataset = PortableDataset.model_validate(payload)
    ranking_report.validate_mode(dataset, "ranking-only")
    with pytest.raises(ValueError, match="quality for reviewed datasets"):
        ranking_report.validate_mode(dataset, "quality")
    payload["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="pending judgments"):
        PortableDataset.model_validate(payload)


def test_pending_ai_assessment_cannot_include_an_approval():
    payload = ai_assisted_dataset_payload()
    payload["review_status"] = "draft"
    payload["queries"][0]["judgments"][0].update(review_status="pending", reviewer_id=None)
    with pytest.raises(ValueError, match="cannot declare human approval"):
        PortableDataset.model_validate(payload)


def test_dataset_rejects_ai_as_reviewer_on_another_judgment():
    payload = ai_assisted_dataset_payload()
    payload["queries"][1]["judgments"][0]["reviewer_id"] = "ai-assessor-fixture"
    with pytest.raises(ValueError, match="declared AI assessor cannot be a reviewer"):
        PortableDataset.model_validate(payload)


@pytest.mark.parametrize("version", [1, True, 3])
def test_ai_assisted_judgments_require_dataset_version_two(version):
    with pytest.raises(ValueError, match="version"):
        PortableDataset.model_validate(ai_assisted_dataset_payload() | {"schema_version": version})


@pytest.mark.parametrize(("section", "field", "value"), [
    ("ai_provenance", "tool", "Another synthetic tool"),
    ("human_review", "approval_reference", "another-synthetic-approval-fixture"),
])
def test_assessment_provenance_changes_dataset_fingerprint(section, field, value):
    payload = ai_assisted_dataset_payload()
    original = run_benchmark.dataset_fingerprint(PortableDataset.model_validate(payload))
    payload["queries"][0]["judgments"][0][section][field] = value
    assert run_benchmark.dataset_fingerprint(PortableDataset.model_validate(payload)) != original


@pytest.mark.parametrize("defect", ["quality", "duplicate", "unknown-protocol", "runtime-key", "query-set", "model"])
def test_report_schema_rejects_inconsistent_reports(defect):
    report = make_report().model_dump(exclude_none=True)
    query = report["results"][0]["queries"][0]
    if defect == "quality":
        query["quality"] = dict.fromkeys(ranking_report.Quality.model_fields, 0.0)
    elif defect == "duplicate":
        query["ranking"][1] = query["ranking"][0]
    elif defect == "unknown-protocol":
        query["ranking"][0]["passage_id"] = logical(0).replace(PROTOCOL, "f" * 64)
    elif defect == "runtime-key":
        query["ranking"][0]["chunk_id"] = 7
    elif defect == "query-set":
        report["results"][1]["queries"].pop()
    else:
        report["reranker"]["repo_id"] = report["embedding"]["repo_id"]
    with pytest.raises(ValueError):
        BenchmarkReport.model_validate(report, context=report_context())


def test_publish_real_report_and_preserve_existing_file(tmp_path):
    destination = tmp_path / "ranking.json"
    report_output.publish(make_report(), destination, report_context())
    previous = destination.read_bytes()
    assert ranking_report.validate_report(previous, report_context()).mode == "ranking-only"
    with pytest.raises(FileExistsError):
        report_output.publish(make_report(), destination, report_context())
    assert destination.read_bytes() == previous


def test_publish_preserves_collision_that_appears_at_publication(tmp_path, monkeypatch):
    destination = tmp_path / "ranking.json"
    validate = report_output.validate_report

    def collision(data, context):
        destination.write_bytes(b"unrelated")
        return validate(data, context)

    monkeypatch.setattr(report_output, "validate_report", collision)
    with pytest.raises(FileExistsError):
        report_output.publish(make_report(), destination, report_context())
    assert destination.read_bytes() == b"unrelated" and list(tmp_path.iterdir()) == [destination]


def test_historical_destination_is_refused():
    with pytest.raises(ValueError, match="historical"):
        report_output.output_destination(CORPUS_DIR.parent / "results" / "new-output.json")


@pytest.fixture
def reranker_files(tmp_path, monkeypatch):
    directory = tmp_path / "weights"
    directory.mkdir()
    manifest = {"schema_version": 1, "repo_id": reranker_assets.MODEL_ID, "revision": reranker_assets.MODEL_REVISION, "files": {}}
    for name in reranker_assets.REQUIRED_FILES:
        content = name.encode()
        (directory / name).write_bytes(content)
        manifest["files"][name] = sha256(content).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reranker_assets, "MANIFEST_PATH", path)
    return directory, path, manifest


@pytest.mark.parametrize("defect", ["corrupt", "missing", "extra", "revision", "manifest-file", "boolean-schema"])
def test_reranker_integrity_and_revision(reranker_files, defect):
    folder, path, manifest = reranker_files
    assert reranker_assets.verified_reranker(folder)[1] == manifest
    if defect == "corrupt":
        (folder / "model.safetensors").write_bytes(b"xxxxx")
    elif defect == "missing":
        (folder / "tokenizer.json").unlink()
    elif defect == "extra":
        (folder / "pytorch_model.bin").write_bytes(b"excluded")
    elif defect == "revision":
        manifest["revision"] = "d" * 40
    elif defect == "manifest-file":
        manifest["files"]["../escape"] = "a" * 64
    else:
        manifest["schema_version"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        reranker_assets.verified_reranker(folder)


def test_prepare_download_requests_only_pinned_assets(reranker_files, tmp_path, monkeypatch):
    folder, _, manifest = reranker_files
    destination = tmp_path / "new"

    def download(**kwargs):
        local = kwargs["local_dir"]
        local.mkdir()
        for file in kwargs["allow_patterns"]:
            (local / file).write_bytes((folder / file).read_bytes())
        (local / ".cache").mkdir()

    download = Mock(side_effect=download)
    monkeypatch.setattr(reranker_assets, "snapshot_download", download)
    reranker_assets.prepare_reranker(destination)
    assert reranker_assets.verified_reranker(destination)[1] == manifest
    assert download.call_args.kwargs["revision"] == reranker_assets.MODEL_REVISION
    assert set(download.call_args.kwargs["allow_patterns"]) == reranker_assets.REQUIRED_FILES
    assert download.call_args.kwargs["token"] is False
    reranker_assets.prepare_reranker(destination)
    download.assert_called_once()


def test_require_ready_never_initializes_empty_seed(monkeypatch):
    connection = Mock()
    connection.exec_driver_sql.return_value.scalar_one.return_value = True
    client = Mock(collection_exists=Mock(return_value=False))
    targets = Mock()
    monkeypatch.setattr(seed.ScriptDirectory, "from_config", Mock(return_value=Mock(get_current_head=Mock(return_value="head"))))
    monkeypatch.setattr(seed, "inspect", Mock(return_value=Mock(get_table_names=Mock(return_value=[]))))
    migrate = Mock()
    monkeypatch.setattr(seed.command, "upgrade", migrate)
    with pytest.raises(SeedError, match="existing seeded"):
        seed.seed_transaction(connection, client, targets, None, {}, require_ready=True)
    migrate.assert_not_called()
    client.create_collection.assert_not_called()


def cli_args(destination, project="aura-benchmark-test"):
    return [
        "--docker-context", "local", "--project", project, "--postgres-port", "15432", "--qdrant-port", "16333",
        "--embedding-model-dir", "unused", "--reranker-model-dir", "unused",
        "--mode", "ranking-only", "--output", str(destination),
    ]


def test_cli_rejects_unverified_target_without_report(tmp_path, capsys):
    destination = tmp_path / "unsafe.json"
    assert run_benchmark.main(cli_args(destination, "production")) == 1
    assert not destination.exists() and "dedicated project" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [
    SeedError("Invalid stored benchmark readiness record"),
    SeedError("Benchmark infrastructure identity changed"),
    RuntimeError("simulated failed execution"),
])
def test_failed_run_does_not_publish_report(tmp_path, monkeypatch, capsys, failure):
    destination = tmp_path / "report.json"
    monkeypatch.setattr(run_benchmark, "resolve_targets", Mock(return_value=Mock()))
    monkeypatch.setattr(run_benchmark, "checked_assets", Mock(return_value=(tmp_path, {})))
    monkeypatch.setattr(run_benchmark, "run_against_seed", Mock(side_effect=failure))
    assert run_benchmark.main(cli_args(destination)) == 1
    assert not destination.exists() and not list(tmp_path.glob(".aura-report-*"))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("changed", ["identity", "record"])
def test_runner_rechecks_seed_before_returning_results(prepared, monkeypatch, changed):
    from contextlib import nullcontext

    record = readiness(prepared)
    observed = []
    target = Mock()
    target.qdrant.return_value = Mock()
    engine = Mock()
    connection = Mock()
    engine.begin.return_value = nullcontext(connection)
    target.engine.return_value = engine

    def checked(sql, client, targets, corpus, model, *, require_ready):
        assert sql is connection and client is target.qdrant.return_value and targets is target
        assert corpus is prepared and require_ready
        observed.append("verified")
        if len(observed) == 3:
            if changed == "identity":
                raise SeedError("Benchmark infrastructure identity changed")
            return object()
        return record

    embedding = Mock()
    reranker = Mock()
    search = Mock()
    evaluate_run = Mock(side_effect=lambda *args: observed.append("evaluated"))
    monkeypatch.setattr(ranking, "Session", lambda **kwargs: nullcontext(object()))
    monkeypatch.setattr(ranking, "seed_transaction", checked)
    monkeypatch.setattr(ranking, "search_service", Mock(return_value=search))
    with pytest.raises(SeedError, match="identity changed|readiness changed"):
        ranking.run_against_seed(target, prepared, {}, embedding, reranker, evaluate_run)
    assert observed == ["verified", "evaluated", "verified"]
    connection.exec_driver_sql.assert_called_once_with("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    engine.dispose.assert_called_once_with()
    target.qdrant.return_value.close.assert_called_once_with()


@pytest.mark.parametrize(("defect", "diagnostic"), [
    ("all-profiles-short", "query IDs"),
    ("missing-profile", "results"),
    ("unexpected-profile", "retrieval configurations"),
    ("duplicate-profile", "retrieval configurations"),
    ("duplicate-query", "query IDs"),
    ("unexpected-query", "query IDs"),
    ("wrong-category", "categories"),
    ("unknown-passage", "unknown logical passage"),
    ("short-ranking", "ranking length"),
    ("wrong-order", "logical tie order"),
    ("nonfinite-score", "finite"),
    ("wrong-corpus", "trusted inputs"),
    ("wrong-dataset", "trusted inputs"),
    ("wrong-model-revision", "model identities"),
    ("empty-quality-field", "quality fields"),
])
def test_validate_report_requires_trusted_catalogs(defect, diagnostic):
    report = make_report().model_dump(exclude_none=True)
    profile = report["results"][0]
    query = profile["queries"][0]
    if defect == "all-profiles-short":
        for item in report["results"]:
            item["queries"].pop()
    elif defect == "missing-profile":
        report["results"].pop()
    elif defect == "unexpected-profile":
        report["results"][-1]["configuration"] = configurations(5, 40, rerank_pools=(5, 10, 25))[-1].model_dump()
    elif defect == "duplicate-profile":
        report["results"][-1] = report["results"][-2]
    elif defect == "duplicate-query":
        profile["queries"][1] = query
    elif defect == "unexpected-query":
        query["query_id"] = "unknown-query"
    elif defect == "wrong-category":
        query["category"] = "software-engineering"
    elif defect == "unknown-passage":
        query["ranking"][0]["passage_id"] = logical(97)
    elif defect == "short-ranking":
        query["ranking"].pop()
    elif defect == "wrong-order":
        query["ranking"].reverse()
    elif defect == "nonfinite-score":
        query["ranking"][0]["score"] = float("inf")
    elif defect == "wrong-corpus":
        report["corpus_fingerprint"] = "f" * 64
    elif defect == "wrong-dataset":
        report["dataset_fingerprint"] = "f" * 64
    elif defect == "wrong-model-revision":
        report["reranker"]["revision"] = "f" * 40
    else:
        query["quality"] = None
    with pytest.raises(ValueError, match=diagnostic):
        ranking_report.validate_report(json.dumps(report), report_context())


def test_report_integrity_requires_context_and_rejects_duplicate_json_keys():
    data = make_report().model_dump_json(exclude_none=True)
    with pytest.raises(ValueError, match="validation context"):
        BenchmarkReport.model_validate_json(data)
    with pytest.raises(ValueError, match="duplicate JSON"):
        ranking_report.validate_report('{"mode":"ranking-only",' + data[1:], report_context())


@pytest.mark.parametrize(("defect", "diagnostic"), [
    ("query", "query metrics"),
    ("category", "category denominator or metrics"),
    ("overall", "overall metrics"),
    ("denominator", "category denominator or metrics"),
    ("category-membership", "categories"),
    ("nonfinite", "finite"),
    ("out-of-range", "less than or equal to"),
])
def test_recomputed_quality_rejects_invalid_report(defect, diagnostic):
    report = make_report(reviewed=True).model_dump(exclude_none=True)
    profile = report["results"][0]
    metrics = profile["queries"][0]["quality"]
    if defect == "query":
        metrics["mrr_at_k"] = 0.125
    elif defect == "category":
        profile["category_quality"]["economics"]["recall_at_k"] = 0.5
    elif defect == "overall":
        profile["quality"]["precision_at_k"] = 0.125
    elif defect == "denominator":
        profile["category_quality"]["economics"]["query_count"] = 2
    elif defect == "category-membership":
        profile["category_quality"]["cybersecurity"] = profile["category_quality"].pop("economics")
    elif defect == "nonfinite":
        metrics["ndcg_at_k"] = float("nan")
    else:
        metrics["mrr_at_k"] = 1.5
    with pytest.raises(ValueError, match=diagnostic):
        ranking_report.validate_report(json.dumps(report), report_context(reviewed=True))


def test_reviewed_report_verifies_hand_calculated_graded_metrics():
    from dataclasses import replace
    from math import log2

    context = report_context(reviewed=True)
    dataset = context.dataset.model_copy(deep=True)
    dataset.queries[0].judgments = [judgment(0, 0), judgment(1, 2), judgment(12, 3)]
    context = replace(context, dataset=dataset)
    payload = make_report(reviewed=True).model_dump(exclude_none=True)
    payload["dataset_fingerprint"] = run_benchmark.dataset_fingerprint(dataset)
    graded = (3 / log2(3)) / (7 + 3 / log2(3))
    metrics = dict(precision_at_k=0.2, recall_at_k=0.5, mrr_at_k=0.5, ndcg_at_k=graded)
    for profile in payload["results"]:
        profile["queries"][0]["quality"] = metrics
        profile["category_quality"]["economics"] = dict(metrics, query_count=1)
        profile["quality"] = {key: value / 2 for key, value in metrics.items()}
    assert ranking_report.validate_report(json.dumps(payload), context).mode == "quality"
    with pytest.raises(ValueError, match="trusted inputs"):
        ranking_report.validate_report(json.dumps(payload), report_context())
    dataset.queries[0].judgments.clear()
    with pytest.raises(ValueError, match="requires unique reviewed"):
        replace(context, dataset=dataset)


def test_report_parent_replacement_cannot_redirect_published_bytes(tmp_path, monkeypatch):
    directory = tmp_path / "reports"
    directory.mkdir()
    renamed = tmp_path / "parked"
    destination = directory / "ranking.json"
    validate = report_output.validate_report

    def replace_parent(data, context):
        verified = validate(data, context)
        try:
            directory.rename(renamed)
        except PermissionError:
            # The Windows directory handles block the attempted rename.
            assert report_output.os.name == "nt"
        else:
            directory.mkdir()
            (directory / "canary").write_bytes(b"unrelated")
        return verified

    monkeypatch.setattr(report_output, "validate_report", replace_parent)
    context = report_context()
    if report_output.os.name == "nt":
        report_output.publish(make_report(), destination, context)
        assert validate(destination.read_bytes(), context)
        assert not renamed.exists()
    else:
        with pytest.raises(OSError, match="changed"):
            report_output.publish(make_report(), destination, context)
        assert not destination.exists() and not (renamed / destination.name).exists()
        assert (directory / "canary").read_bytes() == b"unrelated"


def test_temporary_file_substitution_cannot_change_validated_bytes(tmp_path, monkeypatch):
    destination = tmp_path / "ranking.json"
    validate = report_output.validate_report

    def substitute(data, context):
        verified = validate(data, context)
        names = list(tmp_path.glob(".aura-report-*"))
        if report_output.os.name == "nt":
            assert len(names) == 1
            with pytest.raises(OSError):
                names[0].replace(tmp_path / "replacement")
            with pytest.raises(OSError):
                names[0].write_bytes(b"unvalidated")
        else:
            assert names == []
        return verified

    monkeypatch.setattr(report_output, "validate_report", substitute)
    report = make_report()
    report_output.publish(report, destination, report_context())
    assert destination.read_bytes() == (report.model_dump_json(indent=2, exclude_none=True) + "\n").encode()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("via_failure", ["validation", "publication"])
def test_failed_publication_removes_only_owned_resources(tmp_path, monkeypatch, via_failure):
    destination = tmp_path / "ranking.json"
    canary = tmp_path / ".aura-report-unrelated.tmp"
    canary.write_bytes(b"unrelated")

    def fail(*_):
        raise OSError("synthetic interrupted report")

    if via_failure == "validation":
        monkeypatch.setattr(report_output, "validate_report", fail)
    else:
        api = "windows_destination" if report_output.os.name == "nt" else "posix_destination"
        original = getattr(report_output, api)

        @contextmanager
        def interrupted(path):
            with original(path) as (file, _):
                yield file, fail

        monkeypatch.setattr(report_output, api, interrupted)
    with pytest.raises(OSError, match="synthetic interrupted"):
        report_output.publish(make_report(), destination, report_context())
    assert canary.read_bytes() == b"unrelated"
    assert list(tmp_path.iterdir()) == [canary]


def test_report_directory_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("This test environment cannot create symbolic links")
    with pytest.raises((OSError, ValueError)):
        report_output.publish(make_report(), redirected / "ranking.json", report_context())
    assert not list(real.iterdir())


@pytest.mark.parametrize("defect", ["missing", "corrupt"])
def test_cli_does_not_expose_private_model_locations(tmp_path, monkeypatch, capsys, defect):
    destination = tmp_path / "ranking.json"
    private = tmp_path / "synthetic-private-directory"
    if defect == "corrupt":
        private.mkdir()
        for name in reranker_assets.REQUIRED_FILES:
            (private / name).write_bytes(b"invalid")
    manifest = run_benchmark.run_path(str(run_benchmark.TEST_MODEL_PREPARATION))["load_manifest"]()
    monkeypatch.setattr(run_benchmark, "resolve_targets", Mock(return_value=object()))
    monkeypatch.setattr(run_benchmark, "verified_model", Mock(return_value=(tmp_path, manifest)))
    argv = cli_args(destination)
    argv[argv.index("--reranker-model-dir") + 1] = str(private)
    assert run_benchmark.main(argv) == 1
    outcome = capsys.readouterr()
    assert str(private) not in outcome.err and private.name not in outcome.err
    assert reranker_assets.MODEL_ID in outcome.err
    assert ("checksum mismatch for config.json" if defect == "corrupt" else "model directory is missing") in outcome.err
    assert outcome.out == "" and not destination.exists()


@pytest.mark.parametrize("phase_name", ["resolve_targets", "run_against_seed", "publish"])
def test_cli_does_not_render_nested_credentials(tmp_path, monkeypatch, capsys, phase_name):
    destination = tmp_path / "ranking.json"
    monkeypatch.setattr(run_benchmark, "resolve_targets", Mock(return_value=object()))
    monkeypatch.setattr(run_benchmark, "checked_assets", Mock(return_value=(tmp_path, {})))
    monkeypatch.setattr(run_benchmark, "run_against_seed", Mock(return_value=make_report().results))
    monkeypatch.setattr(run_benchmark, "load_report_context", Mock(return_value=report_context()))
    secret = "postgresql://synthetic:private-token@internal.example/db"

    def fail(*args, **kwargs):
        import sys
        import warnings
        print(secret, file=sys.stderr)
        warnings.warn(secret, stacklevel=1)
        try:
            raise RuntimeError(secret)
        except RuntimeError as error:
            raise ValueError(secret) from error

    monkeypatch.setattr(run_benchmark, phase_name, fail)
    assert run_benchmark.main(cli_args(destination)) == 1
    outcome = capsys.readouterr()
    assert "private-token" not in outcome.err and "internal.example" not in outcome.err
    assert "Portable benchmark failed." in outcome.err and outcome.out == ""
    assert not destination.exists()


def test_metric_comparison_uses_small_absolute_tolerance():
    report = make_report(reviewed=True).model_dump(exclude_none=True)
    metrics = report["results"][0]["queries"][0]["quality"]
    metrics["precision_at_k"] += 5e-13
    context = report_context(reviewed=True)
    assert ranking_report.validate_report(json.dumps(report), context)
    metrics["precision_at_k"] += 2e-12
    with pytest.raises(ValueError, match="query metrics"):
        ranking_report.validate_report(json.dumps(report), context)


def test_cli_publishes_complete_report_against_default_context(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "ranking.json"
    context = run_benchmark.load_report_context("ranking-only")
    logical_ids = sorted(context.passage_ids)
    mapping = SimpleNamespace(points={i: {"passage_id": pid} for i, pid in enumerate(logical_ids)})
    service = SimpleNamespace(search=Mock(return_value=[
        {"chunk_id": pid, "score": 0.5} for pid in logical_ids[:5]
    ]))
    monkeypatch.setattr(run_benchmark, "resolve_targets", Mock(return_value=object()))
    monkeypatch.setattr(run_benchmark, "checked_assets", Mock(return_value=(tmp_path, {})))

    def seeded(targets, corpus, manifest, embedding, reranker, evaluate_run):
        return evaluate_run(service, None, 1, mapping)

    monkeypatch.setattr(run_benchmark, "run_against_seed", seeded)
    assert run_benchmark.main(cli_args(destination)) == 0
    assert capsys.readouterr().err == ""
    report = ranking_report.validate_report(destination.read_bytes(), context)
    assert sum(len(profile.queries) for profile in report.results) == service.search.call_count == 180


def test_cli_does_not_echo_rejected_argument_values(tmp_path, capsys):
    arguments = cli_args(tmp_path / "report.json")
    private = "postgresql://synthetic:private-token@internal.example/db"
    arguments[arguments.index("--postgres-port") + 1] = private
    with pytest.raises(SystemExit) as error:
        run_benchmark.main(arguments)
    assert error.value.code == 2
    assert private not in capsys.readouterr().err


@pytest.mark.skipif(report_output.os.name != "nt", reason="Windows junctions are platform-specific")
def test_report_directory_junction_is_rejected(tmp_path):
    import subprocess

    real = tmp_path / "real"
    real.mkdir()
    redirected = tmp_path / "redirected"
    created = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(redirected), str(real)],
                             capture_output=True, check=False)
    if created.returncode:
        pytest.skip("This test environment cannot create a junction")
    try:
        assert redirected.is_junction()
        with pytest.raises((OSError, ValueError)):
            report_output.publish(make_report(), redirected / "ranking.json", report_context())
        assert not list(real.iterdir())
    finally:
        redirected.rmdir()


def test_report_destination_is_bound_to_preflight_directory(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    parked = tmp_path / "parked"
    output = directory / "ranking.json"
    destination = report_output.output_destination(output)
    directory.rename(parked)
    directory.mkdir()
    with pytest.raises(OSError, match="changed"):
        report_output.publish(make_report(), destination, report_context())
    assert not list(parked.iterdir()) and not list(directory.iterdir())


def parse_report_at(entry_point, data, context):
    if entry_point == "function":
        return ranking_report.validate_report(data, context)
    return BenchmarkReport.model_validate_json(data, context=context)


@pytest.mark.parametrize("entry_point", ["function", "model"])
@pytest.mark.parametrize("conflicting", [False, True], ids=["identical", "conflicting"])
@pytest.mark.parametrize("field", ["dataset_fingerprint", "query_id", "precision_at_k"])
def test_report_json_entry_points_reject_duplicate_keys(entry_point, conflicting, field):
    reviewed = field == "precision_at_k"
    report = make_report(reviewed=reviewed)
    data = report.model_dump_json(exclude_none=True)
    query = report.results[0].queries[0]
    original, alternative = {
        "dataset_fingerprint": (report.dataset_fingerprint, "0" * 64),
        "query_id": (query.query_id, "different-query"),
        "precision_at_k": (query.quality.precision_at_k if reviewed else 0.0, 0.99),
    }[field]
    key = json.dumps(field) + ":"
    token = key + json.dumps(original)
    duplicate = key + json.dumps(alternative if conflicting else original)
    assert token in data
    with pytest.raises(ValueError, match=f"duplicate JSON key '{field}'"):
        parse_report_at(entry_point, data.replace(token, duplicate + "," + token, 1), report_context(reviewed))


@pytest.mark.parametrize("entry_point", ["function", "model"])
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_report_json_entry_points_reject_nonfinite_numbers(entry_point, number):
    data = make_report().model_dump_json(exclude_none=True)
    assert '"score":1.0' in data
    bad = data.replace('"score":1.0', '"score":' + number, 1)
    with pytest.raises(ValueError, match="nonfinite JSON number"):
        parse_report_at(entry_point, bad, report_context())


@pytest.mark.parametrize("entry_point", ["function", "model"])
def test_report_json_entry_points_validate_complete_reports_and_reject_bad_input(entry_point):
    report = make_report()
    data = report.model_dump_json(exclude_none=True).encode()
    context = report_context()
    assert parse_report_at(entry_point, data, context) == report
    with pytest.raises(ValueError, match="validation context"):
        parse_report_at(entry_point, data, None)
    with pytest.raises(json.JSONDecodeError):
        parse_report_at(entry_point, data[:-1], context)


def test_report_generic_json_adapter_fails_closed():
    from pydantic import TypeAdapter

    data = make_report().model_dump_json(exclude_none=True)
    adapter = TypeAdapter(BenchmarkReport)
    with pytest.raises(ValueError, match="Use BenchmarkReport.model_validate_json or validate_report"):
        adapter.validate_json(data, context=report_context())


def test_failed_stream_construction_closes_owned_descriptor_and_recovers(tmp_path, monkeypatch):
    import errno

    folder = tmp_path / "reports"
    folder.mkdir()
    canary = folder / "keep.json"
    canary.write_bytes(b"unrelated")
    target = folder / "report.json"
    os = report_output.os
    original_open, original_fdopen = os.open, os.fdopen
    opened, closed, received = [], [], []
    report, context = make_report(), report_context()

    def track_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail_fdopen(fd, *args, **kwargs):
        received.append(fd)
        # Simulate a late constructor failure after a partial wrapper was closed.
        with original_fdopen(fd, *args, **kwargs) as stream:
            assert stream.fileno() == fd
        os.fstat(fd)
        raise OSError("injected stream initialization failure")

    with monkeypatch.context() as patch:
        if os.name != "nt":
            patch.setattr(os, "open", track_open)
        patch.setattr(os, "close", Mock(side_effect=os.close))
        patch.setattr(os, "fdopen", fail_fdopen)
        with pytest.raises(OSError, match="injected stream initialization failure") as error:
            report_output.publish(report, target, context)
        closed.extend(call.args[0] for call in os.close.call_args_list)
    assert not getattr(error.value, "__notes__", ())
    assert len(received) == 1
    assert sorted(closed) == sorted(opened or received)
    for fd in opened or received:
        with pytest.raises(OSError) as error:
            os.fstat(fd)
        assert error.value.errno == errno.EBADF
    moved = tmp_path / "renamed"
    folder.rename(moved)
    assert sorted(p.name for p in moved.iterdir()) == ["keep.json"]
    assert (moved / "keep.json").read_bytes() == b"unrelated"
    target = moved / target.name
    report_output.publish(report, target, context)
    expected = target.read_bytes()
    assert ranking_report.validate_report(expected, context) == report
    with pytest.raises(FileExistsError):
        report_output.publish(report, target, context)
    assert target.read_bytes() == expected


@pytest.mark.skipif(report_output.os.name != "nt", reason="Windows uses native handle-to-descriptor conversion")
def test_windows_failed_descriptor_conversion_closes_native_handle(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    folder = tmp_path / "reports"
    folder.mkdir()
    target = folder / "report.json"
    canary = folder / "keep.json"
    canary.write_bytes(b"unrelated")
    captured, closed = [], []
    report, context = make_report(), report_context()
    load_library = ctypes.WinDLL
    kernel = load_library("kernel32", use_last_error=True)
    real_close = kernel.CloseHandle
    real_close.argtypes = [wintypes.HANDLE]
    real_close.restype = wintypes.BOOL

    def record_close(handle):
        closed.append(handle)
        return real_close(handle)

    api = SimpleNamespace(
        CreateFileW=kernel.CreateFileW,
        CloseHandle=Mock(side_effect=record_close),
        SetFileInformationByHandle=kernel.SetFileInformationByHandle,
    )

    def fail_conversion(handle, _flags):
        captured.append(handle)
        raise OSError("injected handle conversion failure")

    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "WinDLL", lambda name, **kw: api if name == "kernel32" else load_library(name, **kw))
        patch.setattr(msvcrt, "open_osfhandle", fail_conversion)
        with pytest.raises(OSError, match="injected handle conversion failure") as error:
            report_output.publish(report, target, context)
    assert not getattr(error.value, "__notes__", ())
    assert len(captured) == 1 and closed[0] == captured[0]
    assert len(closed) == len(set(closed)) == len(target.parents) + 1
    moved = tmp_path / "renamed"
    folder.rename(moved)
    assert [p.name for p in moved.iterdir()] == ["keep.json"]
    assert (moved / "keep.json").read_bytes() == b"unrelated"


def test_stream_initialization_error_is_preserved_if_cleanup_also_fails(tmp_path, monkeypatch):
    os = report_output.os
    target = tmp_path / "report.json"
    original_close = os.close
    staging = []

    def fail_open(fd, *_args, **_kwargs):
        staging.append(fd)
        raise OSError("primary initialization failure")

    def fail_after_close(fd):
        original_close(fd)
        if fd in staging:
            raise OSError("secondary cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fdopen", fail_open)
        patch.setattr(os, "close", fail_after_close)
        with pytest.raises(OSError, match="primary initialization failure") as error:
            report_output.publish(make_report(), target, report_context())
    assert any("secondary cleanup failure" in note for note in error.value.__notes__)
    assert not list(tmp_path.iterdir())
