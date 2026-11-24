import json
import subprocess
import sys
from copy import deepcopy

import pytest

from evaluation.check_retrieval_regression import BASELINE_PATH, compare_results


@pytest.fixture
def reports():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return baseline, deepcopy(baseline)


def test_matches_saved_baseline_by_id_and_ignores_latency(reports):
    baseline, current = reports
    current["retrieval_configs"].reverse()
    for result in current["results"].values():
        result["mean_latency_ms"] *= 10
        result["per_query"].reverse()
        for query in result["per_query"]:
            query["latency_ms"] *= 10

    assert compare_results(baseline, current, 0, 0, {}) == []


@pytest.mark.parametrize("metric", ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"])
def test_aggregate_tolerance_is_inclusive_and_applies_to_each_metric(reports, metric):
    baseline, current = reports
    baseline_score = baseline["results"]["hybrid"][metric]
    current["results"]["hybrid"][metric] = baseline_score - 0.08

    assert compare_results(baseline, current, 0.08, 0, {}) == []
    assert compare_results(baseline, current, 0.07, 0, {}) == [
        f"hybrid / {metric}: {baseline_score:.6f} -> {baseline_score - 0.08:.6f} "
        "(drop 0.080000, allowed 0.070000)"
    ]


def test_query_drop_detected_when_aggregate_scores_match(reports):
    baseline, current = reports
    query = current["results"]["hybrid"]["per_query"][0]
    query["ndcg_at_5"] -= 0.15

    failures = compare_results(baseline, current, 0.25, 0.1, {})
    assert len(failures) == 1
    assert f"hybrid / {query['query_id']} / ndcg_at_5" in failures[0]
    assert "drop 0.150000, allowed 0.100000" in failures[0]
    assert compare_results(baseline, current, 0, 0.15, {}) == []


def test_improvement_passes_but_can_still_miss_absolute_floor(reports):
    baseline, current = reports
    current["results"]["dense"]["mrr"] = 0.89

    assert compare_results(baseline, current, 0, 0, {}) == []
    assert compare_results(baseline, current, 0, 0, {"mrr": 0.9}) == [
        "dense / mrr: 0.890000 is below the minimum 0.900000"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", "Different benchmark"),
        ("dataset_version", "0.3"),
        ("workspace_id", 2),
        ("top_k", 10),
        ("query_count", 11),
    ],
)
def test_rejects_mismatched_dataset_metadata(reports, field, value):
    baseline, current = reports
    current[field] = value

    with pytest.raises(ValueError, match=rf"{field} differs: baseline .+, current"):
        compare_results(baseline, current, 0, 0, {})


def test_rejects_changed_retrieval_settings(reports):
    baseline, current = reports
    current_config = next(
        config for config in current["retrieval_configs"] if config["name"] == "hybrid_rerank_5"
    )
    current_config["candidate_pool_size"] = 7

    with pytest.raises(ValueError, match=r"hybrid_rerank_5: retrieval settings differ"):
        compare_results(baseline, current, 0, 0, {})


def test_rejects_inconsistent_result_metadata(reports):
    baseline, current = reports
    current["results"]["dense"]["mode"] = "hybrid"

    with pytest.raises(ValueError, match=r"current / dense: result mode"):
        compare_results(baseline, current, 0, 0, {})


def test_rejects_missing_configuration(reports):
    baseline, current = reports
    del current["results"]["dense"]

    with pytest.raises(ValueError, match=r"current result configurations differ.*missing: .*dense"):
        compare_results(baseline, current, 0, 0, {})


def test_identifies_missing_and_unexpected_queries(reports):
    baseline, current = reports
    query = current["results"]["dense"]["per_query"][0]
    original = query["query_id"]
    query["query_id"] = "replacement"

    pattern = rf"dense query IDs differ.*missing: .*{original}.*unexpected: .*replacement"
    with pytest.raises(ValueError, match=pattern):
        compare_results(baseline, current, 0, 0, {})


def test_rejects_duplicate_queries(reports):
    baseline, current = reports
    queries = current["results"]["dense"]["per_query"]
    queries[1]["query_id"] = queries[0]["query_id"]

    with pytest.raises(ValueError, match=r"current / dense: invalid or duplicate query_id"):
        compare_results(baseline, current, 0, 0, {})


def test_rejects_query_category_change(reports):
    baseline, current = reports
    query = current["results"]["dense"]["per_query"][0]
    query["category"] = "different"

    with pytest.raises(ValueError, match=rf"dense / {query['query_id']}: query category differs"):
        compare_results(baseline, current, 0, 0, {})


@pytest.mark.parametrize("value", [None, "0.5", True, -0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_ranking_scores(reports, value):
    baseline, current = reports
    current["results"]["dense"]["mrr"] = value

    with pytest.raises(ValueError, match=r"dense: mrr must be a number between 0 and 1"):
        compare_results(baseline, current, 0, 0, {})


def run_checker(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "evaluation.check_retrieval_regression", *map(str, arguments)],
        cwd=BASELINE_PATH.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_passes_with_default_baseline_and_fails_after_regression(tmp_path, reports):
    _, current = reports
    report = tmp_path / "current.json"
    report.write_text(json.dumps(current), encoding="utf-8")

    passed = run_checker(report)
    assert passed.returncode == 0, passed.stderr
    assert "Retrieval quality check passed" in passed.stdout

    current["results"]["dense"]["mrr"] -= 0.125
    report.write_text(json.dumps(current), encoding="utf-8")
    failed = run_checker(report, "--baseline", BASELINE_PATH, "--aggregate-tolerance", "0.1")
    assert failed.returncode == 1, failed.stderr
    assert "dense / mrr" in failed.stdout
    assert "drop 0.125000, allowed 0.100000" in failed.stdout


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("{", "Expecting property name"),
        ("[]", "expected a JSON object"),
        ("{}", "missing required field 'dataset_name'"),
    ],
)
def test_cli_returns_input_error_for_malformed_report(tmp_path, content, expected):
    path = tmp_path / "current.json"
    path.write_text(content, encoding="utf-8")

    result = run_checker(path)
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("report_role", ["baseline", "candidate"])
@pytest.mark.parametrize(
    ("original", "with_duplicate", "key"),
    [
        ('"mrr": 1.0', '"mrr": 0.0, "mrr": 1.0', "mrr"),
        ('"mrr": 1.0', '"mrr": 1.0, "mrr": 1.0', "mrr"),
        ('"dataset_version":', '"dataset_version": "invalid", "dataset_version":', "dataset_version"),
        (
            '"retrieval_configs": [{"name": "dense", "mode":',
            '"retrieval_configs": [{"name": "dense", "mode": "invalid", "mode":',
            "mode",
        ),
        (
            '"per_query": [{"query_id":',
            '"per_query": [{"query_id": "invalid", "query_id":',
            "query_id",
        ),
    ],
    ids=["conflicting-metric", "identical-metric", "metadata", "configuration", "per-query"],
)
def test_cli_rejects_duplicate_report_keys(
    tmp_path, reports, report_role, original, with_duplicate, key
):
    report_json = json.dumps(reports[0])
    assert original in report_json
    paths = {name: tmp_path / f"{name}.json" for name in ("baseline", "candidate")}
    for path in paths.values():
        path.write_text(report_json, encoding="utf-8")

    malformed = paths[report_role]
    malformed.write_text(report_json.replace(original, with_duplicate, 1), encoding="utf-8")
    result = run_checker(paths["candidate"], "--baseline", paths["baseline"])

    assert result.returncode == 2, result.stdout + result.stderr
    assert f"{malformed}: duplicate JSON key {key!r}" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_parses_valid_baseline_and_candidate_reports(tmp_path, reports):
    baseline, current = reports
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    result = run_checker(current_path, "--baseline", baseline_path)

    assert result.returncode == 0, result.stderr
    assert "Retrieval quality check passed." in result.stdout


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--aggregate-tolerance", "nan"], "expected a number between 0 and 1"),
        (["--query-tolerance", "1.1"], "expected a number between 0 and 1"),
        (["--minimum", "latency=0.5"], "expected METRIC=VALUE"),
        (["--minimum", "mrr=0.8", "--minimum", "mrr=0.9"], "each minimum at most once"),
    ],
)
def test_cli_rejects_invalid_thresholds(args, expected):
    result = run_checker(BASELINE_PATH, *args)
    assert result.returncode == 2
    assert expected in result.stderr


def test_cli_rejects_using_baseline_as_current_report():
    result = run_checker(BASELINE_PATH)
    assert result.returncode == 2
    assert "baseline and current results must be separate files" in result.stderr
