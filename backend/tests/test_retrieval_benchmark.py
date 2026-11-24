import json
from contextlib import nullcontext

from evaluation import run_retrieval_benchmark as benchmark
from evaluation.check_retrieval_regression import BASELINE_PATH, compare_results


def test_benchmark_cli_writes_comparable_report_to_separate_file(tmp_path, monkeypatch):
    baseline_bytes = BASELINE_PATH.read_bytes()
    baseline = json.loads(baseline_bytes)
    recorded_rankings = {
        (config["mode"], config["candidate_pool_size"], query["query_id"]): query["retrieved_chunk_ids"]
        for config in baseline["results"].values()
        for query in config["per_query"]
    }
    dataset = benchmark.load_dataset()
    query_ids = {query["query"]: query["query_id"] for query in dataset["queries"]}

    class RecordedSearch:
        def search(self, *, db, workspace_id, query, limit, mode, candidate_pool_size=None):
            assert workspace_id == dataset["workspace_id"]
            assert limit == baseline["top_k"]
            ranking = recorded_rankings[mode, candidate_pool_size, query_ids[query]]
            return [{"chunk_id": chunk_id} for chunk_id in ranking]

    monkeypatch.setattr(benchmark, "SearchService", RecordedSearch)
    monkeypatch.setattr(benchmark, "SessionLocal", nullcontext)
    output = tmp_path / "reports" / "current.json"

    benchmark.main(["--output", str(output)])

    assert BASELINE_PATH.read_bytes() == baseline_bytes
    assert compare_results(baseline, json.loads(output.read_bytes()), 0, 0, {}) == []


def test_benchmark_cli_keeps_default_output_path(monkeypatch):
    output_paths = []
    monkeypatch.setattr(benchmark, "run_benchmark", output_paths.append)

    benchmark.main([])

    assert output_paths == [BASELINE_PATH]
