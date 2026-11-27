from copy import deepcopy
from io import StringIO
import json
import math
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from evaluation.portable_benchmark import check_quality_regression as gate
from evaluation.portable_benchmark.corpus import JSONNumberUnderflow, read_json
from evaluation.portable_benchmark.ranking_report import CategoryQuality, Quality, mean_quality, quality_scores, validate_report


@pytest.fixture(scope="module")
def context():
    return gate.trusted_context()


@pytest.fixture(scope="module")
def inputs(context):
    baseline = validate_report(gate.BASELINE_PATH.read_bytes(), context)
    policy = gate.Policy.model_validate(gate.decode_json(gate.POLICY_PATH.read_bytes()))
    return baseline, policy


def reviewed_scores(context, query_id, ids):
    query = next(query for query in context.dataset.queries if query.query_id == query_id)
    return quality_scores(ids, query.judgments, context.top_k)


def rebuild_candidate(baseline, context, profile, replacements):
    data = baseline.model_dump(mode="json")
    group = next(group for group in data["results"] if group["configuration"]["name"] == profile)
    categories = {}
    all_scores = []
    for query in group["queries"]:
        ids = replacements.get(query["query_id"])
        if ids is not None:
            query["ranking"] = [
                {"passage_id": pid, "score": float(len(ids) - rank)} for rank, pid in enumerate(ids)
            ]
        scores = reviewed_scores(context, query["query_id"], [row["passage_id"] for row in query["ranking"]])
        query["quality"] = scores.model_dump()
        categories.setdefault(query["category"], []).append(scores)
        all_scores.append(scores)
    group["quality"] = mean_quality(all_scores)
    group["category_quality"] = {
        category: CategoryQuality(query_count=len(rows), **mean_quality(rows)).model_dump()
        for category, rows in categories.items()
    }
    return validate_report(json.dumps(data).encode(), context)


def ranked_ids(report, profile, query_id):
    group = next(group for group in report.results if group.configuration.name == profile)
    query = next(query for query in group.queries if query.query_id == query_id)
    return [item.passage_id for item in query.ranking]


def judgment_map(context, query_id):
    query = next(query for query in context.dataset.queries if query.query_id == query_id)
    return {judgment.passage_id: judgment.grade for judgment in query.judgments}


def replace_hit(report, context, profile, query_id, new_grade):
    ids = ranked_ids(report, profile, query_id)
    grades = judgment_map(context, query_id)
    replacement = next(pid for pid in sorted(grades) if pid not in ids and grades[pid] == new_grade)
    variants = [
        ids[:index] + [replacement] + ids[index + 1:]
        for index, pid in enumerate(ids) if grades[pid] > new_grade
    ]
    return min(variants, key=lambda ranking: reviewed_scores(context, query_id, ranking).ndcg_at_k)


def swap_first_positive(report, context, profile, query_id, metric):
    ids = ranked_ids(report, profile, query_id)
    grades = judgment_map(context, query_id)
    first = next(index for index, pid in enumerate(ids) if grades[pid] > 0)
    variants = []
    for later in range(first + 1, len(ids)):
        ranking = ids.copy()
        ranking[first], ranking[later] = ranking[later], ranking[first]
        variants.append(ranking)
    return min(variants, key=lambda ranking: getattr(reviewed_scores(context, query_id, ranking), metric))


def failing_candidate(baseline, context):
    profile = "dense"
    edits = {qid: replace_hit(baseline, context, profile, qid, 0) for qid in ("cy-03", "econ-08", "cy-07", "ai-04")}
    return rebuild_candidate(baseline, context, profile, edits)


def write_candidate(path, candidate):
    path.write_text(candidate.model_dump_json(), encoding="utf-8")
    return path


def test_baseline_against_itself(inputs, context, capsys):
    result = gate.compare_reports(inputs[0], inputs[0], context, inputs[1])
    assert len(result["quality_checks"]) == 20 and len(result["categories"]) == 25
    assert all(check["drop"] == 0 and check["decision"] == "PASS" for check in result["quality_checks"])
    assert result["verdict"] == "PASS" and result["warning_count"] == 0
    assert not result["query_changes"] and not result["ranking_changes"]
    assert gate.main([str(gate.BASELINE_PATH)]) == 0
    assert "PASS: 0 blocking failures, 0 warnings." in capsys.readouterr().out


def test_reproduction_has_identical_quality_and_nonblocking_latency(capsys):
    reproduction = gate.BASELINE_PATH.with_name("portable_quality_v1_run2.json")
    args = [str(reproduction), "--format", "json"]
    assert gate.main(args) == 0
    first = capsys.readouterr().out
    assert gate.main(args) == 0
    assert capsys.readouterr().out == first
    result = json.loads(first)
    assert result["candidate_sha256"] == "fd44df2736b9c22d6cd904cfe4499a48105deb66ccc5a88db86f45aeccf8c4d5"
    assert all(check["drop"] == 0 for check in result["quality_checks"])
    assert result["changed_ordered_rankings"] == result["warning_count"] == 0
    assert any(row["change_ms"] != 0 for profile in result["latency"] for row in profile["comparisons"])


def test_valid_improvement_passes(inputs, context):
    baseline, policy = inputs
    grades = judgment_map(context, "sw-03")
    ideal = sorted(grades, key=lambda pid: (-grades[pid], pid))[:5]
    candidate = rebuild_candidate(baseline, context, "dense", {"sw-03": ideal})
    result = gate.compare_reports(baseline, candidate, context, policy)
    assert result["verdict"] == "PASS"
    assert result["query_changes"][0]["query_id"] == "sw-03"
    assert any(check["drop"] < 0 for check in result["quality_checks"])


def test_single_hit_loss_within_tolerance_and_changes_diagnosed(inputs, context):
    baseline, policy = inputs
    ids = replace_hit(baseline, context, "dense", "cy-03", 0)
    candidate = rebuild_candidate(baseline, context, "dense", {"cy-03": ids})
    result = gate.compare_reports(baseline, candidate, context, policy)
    query = result["query_changes"][0]
    assert result["verdict"] == "PASS" and result["changed_ordered_rankings"] == 1
    assert (query["profile"], query["query_id"], query["category"]) == ("dense", "cy-03", "cybersecurity")
    assert next(x["drop"] for x in query["metrics"] if x["metric"] == "precision_at_k") == pytest.approx(.2)
    changes = result["ranking_changes"][0]
    assert len(changes["added"]) == len(changes["removed"]) == 1
    assert changes["score_changes"] and not changes["moved"]


@pytest.mark.parametrize(("drop", "tolerance", "expected"), [
    (.020, .020, False),
    (.020 + gate.EPSILON, .020, False),
    (math.nextafter(.020 + gate.EPSILON, math.inf), .020, True),
    (.050 + gate.EPSILON, .050, False),
    (math.nextafter(.050 + gate.EPSILON, math.inf), .050, True),
])
def test_inclusive_numerical_boundary(drop, tolerance, expected):
    assert gate.exceeds_tolerance(drop, tolerance, gate.EPSILON) is expected


@pytest.mark.parametrize(("metric", "edits"), [
    ("precision_at_k", "four_losses"),
    ("recall_at_k", "three_losses"),
    ("mrr_at_k", "demote"),
    ("ndcg_at_k", "marginal"),
])
def test_each_metric_independently_blocks_valid_rankings(inputs, context, metric, edits):
    baseline, policy = inputs
    if edits in ("four_losses", "three_losses"):
        queries = ("cy-03", "econ-08", "cy-07", "ai-04") if edits == "four_losses" else ("cy-03", "econ-08", "cy-07")
        replacements = {qid: replace_hit(baseline, context, "dense", qid, 0) for qid in queries}
    elif edits == "demote":
        replacements = {"cy-06": swap_first_positive(baseline, context, "dense", "cy-06", metric)}
    else:
        replacements = {"cy-06": replace_hit(baseline, context, "dense", "cy-06", 1)}
    candidate = rebuild_candidate(baseline, context, "dense", replacements)
    specific = policy.model_copy(update={"blocking_tolerances": Quality(
        **{key: getattr(policy.blocking_tolerances, key) if key == metric else 1.0 for key in gate.METRICS}
    )})
    result = gate.compare_reports(baseline, candidate, context, specific)
    assert [(x["profile"], x["metric"]) for x in result["quality_checks"] if x["decision"] == "FAIL"] == [("dense", metric)]


def test_valid_failure_cannot_be_compensated_by_another_profile(inputs, context, tmp_path, capsys):
    baseline, policy = inputs
    candidate = failing_candidate(baseline, context)
    grades = judgment_map(context, "cy-03")
    ideal = sorted(grades, key=lambda pid: (-grades[pid], pid))[:5]
    candidate = rebuild_candidate(candidate, context, "hybrid", {"cy-03": ideal})
    result = gate.compare_reports(baseline, candidate, context, policy)
    assert result["verdict"] == "FAIL"
    assert {x["profile"] for x in result["quality_checks"] if x["decision"] == "FAIL"} == {"dense"}
    assert any(x["drop"] < 0 for x in result["quality_checks"] if x["profile"] == "hybrid")
    assert gate.main([str(write_candidate(tmp_path / "failed.json", candidate)), "--format=json"]) == 1
    assert json.loads(capsys.readouterr().out)["blocking_failure_count"] == result["blocking_failure_count"]
    assert result["query_changes"] == sorted(result["query_changes"], key=lambda row: (
        -max(0, *(change["drop"] for change in row["metrics"])), row["profile"], row["query_id"],
    ))


def test_cross_domain_warning_is_nonblocking_and_has_moved_passages(inputs, context, tmp_path, capsys):
    baseline, _ = inputs
    ids = swap_first_positive(baseline, context, "hybrid", "cross-04", "ndcg_at_k")
    candidate = rebuild_candidate(baseline, context, "hybrid", {"cross-04": ids})
    assert gate.main([str(write_candidate(tmp_path / "warning.json", candidate)), "--format=json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [(x["profile"], x["category"], x["query_count"]) for x in result["categories"] if x["warning"]] == [
        ("hybrid", "cross-domain", 4)
    ]
    movement = result["ranking_changes"][0]
    assert movement["ordered_ranking_changed"] and len(movement["moved"]) == 2
    assert not movement["added"] and not movement["removed"]


def test_comparison_uses_recomputed_quality_even_within_validation_tolerance(inputs, context):
    baseline, policy = inputs
    data = baseline.model_dump(mode="json")
    data["results"][0]["quality"]["precision_at_k"] -= 5e-13
    candidate = validate_report(json.dumps(data).encode(), context)
    result = gate.compare_reports(baseline, candidate, context, policy)
    assert result["quality_checks"][0]["drop"] == 0


def test_new_runtime_and_score_only_changes_are_diagnostic(inputs, context):
    baseline, policy = inputs
    data = baseline.model_dump(mode="json")
    data["runtime"]["system"] = "another-platform"
    for item in data["results"][0]["queries"][0]["ranking"]:
        item["score"] += .125
    candidate = validate_report(json.dumps(data).encode(), context)
    result = gate.compare_reports(baseline, candidate, context, policy)
    assert result["verdict"] == "PASS" and result["changed_ordered_rankings"] == 0
    assert not result["query_changes"]
    assert len(result["ranking_changes"]) == 1 and len(result["ranking_changes"][0]["score_changes"]) == 5


def test_latency_zero_large_values_and_inconsistent_summaries_are_nonblocking(inputs, context):
    baseline, policy = inputs
    base_data = baseline.model_dump(mode="json")
    for query in base_data["results"][0]["queries"]:
        query["latency_ms"] = 0
    base_data["results"][0]["latency"] = dict.fromkeys(("mean_ms", "median_ms", "p95_ms"), 0.0)
    base = validate_report(json.dumps(base_data).encode(), context)
    candidate_data = deepcopy(base_data)
    for query in candidate_data["results"][0]["queries"]:
        query["latency_ms"] = 1e308
    candidate = validate_report(json.dumps(candidate_data).encode(), context)
    result = gate.compare_reports(base, candidate, context, policy)
    assert result["verdict"] == "PASS" and result["warning_count"] == 3
    assert all(row["change_ms"] == 1e308 and row["change_percent"] is None for row in result["latency"][0]["comparisons"])
    assert len(result["latency"][0]["inconsistent_summaries"]) == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("defect", [
    "missing query", "duplicate query", "unexpected query", "missing profile", "duplicate profile",
    "retrieval settings", "unknown passage",
    "duplicate passage", "dataset", "corpus", "model", "protocol", "draft mode", "aggregate", "invalid latency",
])
def test_invalid_candidate_report_returns_exit_two(inputs, tmp_path, capsys, defect):
    baseline, _ = inputs
    data = baseline.model_dump(mode="json")
    group = data["results"][0]
    if defect == "missing query":
        group["queries"].pop()
    elif defect == "duplicate query":
        group["queries"][1] = group["queries"][0]
    elif defect == "unexpected query":
        group["queries"][0]["query_id"] = "unknown-query"
    elif defect == "missing profile":
        data["results"].pop()
    elif defect == "duplicate profile":
        data["results"][1] = group
    elif defect == "retrieval settings":
        data["results"][1]["configuration"]["candidate_pool_size"] = 21
    elif defect == "unknown passage":
        item = group["queries"][0]["ranking"][0]
        item["passage_id"] = item["passage_id"].rsplit(":", 1)[0] + ":" + "0" * 64
    elif defect == "duplicate passage":
        ranking = group["queries"][0]["ranking"]
        ranking[1] = ranking[0]
    elif defect in ("dataset", "corpus"):
        data[f"{defect}_fingerprint"] = "0" * 64
    elif defect == "model":
        data["embedding"]["manifest_sha256"] = "0" * 64
    elif defect == "protocol":
        data["evaluation_protocol"] = "another-protocol"
    elif defect == "draft mode":
        data["mode"] = "ranking-only"
    elif defect == "aggregate":
        group["quality"]["precision_at_k"] += .1
    else:
        group["queries"][0]["latency_ms"] = -1
    path = tmp_path / "private-candidate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert gate.main([str(path), "--format=json"]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error"]["phase"] == "candidate"
    assert str(tmp_path) not in output and path.name not in output


@pytest.mark.parametrize(("raw", "message"), [
    (b'{"private":', "Invalid JSON"),
    (b'{"private":1,"private":2}', "Duplicate JSON key"),
    (b'{"private":1e400}', "Nonfinite JSON number"),
])
def test_invalid_json_output_does_not_expose_input(tmp_path, capsys, raw, message):
    path = tmp_path / "private.json"
    path.write_bytes(raw)
    assert gate.main([str(path), "--format=json"]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error"]["message"].startswith(message)
    assert "private" not in output


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", 2), ("schema_version", True), ("baseline_sha256", "0" * 64),
    ("epsilon", 0), ("epsilon", .1), ("epsilon", True), ("epsilon", float("inf")),
    ("category_ndcg_warning", True), ("category_ndcg_warning", -1),
    ("category_ndcg_warning", 1.1), ("category_ndcg_warning", float("nan")),
    ("blocking_tolerances", {"precision_at_k": .02}),
    ("blocking_tolerances", {"precision_at_k": .02, "recall_at_k": .03, "mrr_at_k": .02, "rr_at_k": .02}),
    ("extra", "unsupported"),
])
def test_invalid_policy_schema(inputs, field, value):
    policy = inputs[1].model_dump()
    policy[field] = value
    with pytest.raises(ValidationError):
        gate.Policy.model_validate(policy)


@pytest.mark.parametrize("value", [True, -.01, 1.1, float("inf")])
def test_invalid_blocking_tolerances(inputs, value):
    policy = inputs[1].model_dump()
    policy["blocking_tolerances"]["precision_at_k"] = value
    with pytest.raises(ValidationError):
        gate.Policy.model_validate(policy)


def test_policy_duplicate_keys_fail_without_exposing_private_key(tmp_path, monkeypatch, capsys):
    policy = tmp_path / "policy.json"
    policy.write_bytes(b'{"private": 1, "private": 2}')
    monkeypatch.setattr(gate, "POLICY_PATH", policy)
    assert gate.main([str(gate.BASELINE_PATH), "--format=json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == {"phase": "policy", "message": "Duplicate JSON key."}


def test_baseline_crlf_only_normalization_and_pinned_identity(tmp_path, monkeypatch, capsys):
    raw = gate.BASELINE_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert gate.approved_baseline_bytes(raw.replace(b"\n", b"\r\n")) == raw
    changed = tmp_path / "wrong-baseline.json"
    changed.write_bytes(raw + b" ")
    monkeypatch.setattr(gate, "BASELINE_PATH", changed)
    assert gate.main([str(changed), "--format=json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["phase"] == "baseline"


def test_cli_does_not_write_input_files(inputs, context, tmp_path, capsys):
    candidate = write_candidate(tmp_path / "candidate.json", failing_candidate(inputs[0], context))
    files = (candidate, gate.BASELINE_PATH, gate.POLICY_PATH)
    original = [path.read_bytes() for path in files]
    assert gate.main([str(candidate)]) == 1
    assert "FAIL:" in capsys.readouterr().out
    assert [path.read_bytes() for path in files] == original


def test_json_argument_errors_are_structured(capsys):
    assert gate.main(["--format", "json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["phase"] == "arguments"


@pytest.mark.parametrize("token", ["-1e-999", "-100E-1001", "-0.01e-997", "1e-999", "1e-9999999999999999999999"])
def test_shared_json_decoders_reject_underflow_for_both_signs(tmp_path, token):
    payload = f'{{"value": {token}}}'.encode("ascii")
    path = tmp_path / "input.json"
    path.write_bytes(payload)
    with pytest.raises(JSONNumberUnderflow):
        gate.decode_json(payload)
    with pytest.raises(JSONNumberUnderflow, match="input.json"):
        read_json(path)


@pytest.mark.parametrize(("token", "expected"), [
    ("0", 0), ("0e-999", 0.0), ("-0.0E-999", -0.0), ("0e+9999999999999999999999", 0.0),
    ("5e-324", math.ulp(0.0)), ("-5e-324", -math.ulp(0.0)), ("0.5", .5), ("-0.5", -.5),
])
def test_shared_json_keeps_exact_zero_subnormals_and_signed_finite_values(token, expected):
    value = gate.decode_json(f'{{"value": {token}}}')["value"]
    assert value == expected
    assert math.copysign(1, value) == math.copysign(1, expected)


@pytest.mark.parametrize(("raw", "reason"), [
    ('{"value": 1e999}', "nonfinite JSON number"),
    ('{"value": -1e999}', "nonfinite JSON number"),
    ('{"value": NaN}', "nonfinite JSON number"),
    ('{"value":0,"value":0}', "duplicate JSON key"),
])
def test_shared_json_existing_rejections(raw, reason):
    with pytest.raises(ValueError, match=reason):
        gate.decode_json(raw)


@pytest.mark.parametrize("token", ["-1e-999", "-100E-1001", "1e-999"])
def test_raw_candidate_latency_underflow_returns_exit_two(inputs, tmp_path, capsys, token):
    baseline, _ = inputs
    data = baseline.model_dump(mode="json")
    data["results"][0]["queries"][0]["latency_ms"] = 1.234567890123
    raw = json.dumps(data)
    assert raw.count("1.234567890123") == 1
    candidate = tmp_path / "candidate.json"
    candidate.write_text(raw.replace("1.234567890123", token), encoding="ascii")
    assert gate.main([str(candidate), "--format=json"]) == 2
    streams = capsys.readouterr()
    assert not streams.err
    assert json.loads(streams.out)["error"] == {
        "phase": "candidate", "message": "Nonzero JSON number underflows to zero.",
    }


@pytest.mark.parametrize(("token", "expected"), [
    ("-1e-999", 2), ("1e-999", 2), ("-0.0e-999", 0), ("5e-324", 0), ("-5e-324", 2),
])
def test_raw_policy_tolerance_preserves_numeric_range_checks(tmp_path, monkeypatch, capsys, token, expected):
    path = tmp_path / "policy.json"
    approved = gate.POLICY_PATH.read_bytes()
    literal = b'"precision_at_k": 0.02'
    assert literal in approved
    path.write_bytes(approved.replace(literal, b'"precision_at_k": ' + token.encode()))
    monkeypatch.setattr(gate, "POLICY_PATH", path)
    assert gate.main([str(gate.BASELINE_PATH), "--format=json"]) == expected
    streams = capsys.readouterr()
    assert not streams.err
    result = json.loads(streams.out)
    if expected:
        assert result["error"]["phase"] == "policy" and result["verdict"] == "ERROR"
        if token.endswith("999"):
            assert result["error"]["message"] == "Nonzero JSON number underflows to zero."
    else:
        assert result["verdict"] == "PASS"


class FaultyOutput(StringIO):
    def __init__(self, phase, error_type=OSError):
        super().__init__()
        self.phase = phase
        self.error_type = error_type
        self.write_attempts = 0
        self.flush_attempts = 0

    def write(self, text):
        self.write_attempts += 1
        if self.phase == "write":
            raise self.error_type("synthetic private output path")
        return super().write(text)

    def flush(self):
        self.flush_attempts += 1
        if self.phase == "flush":
            raise self.error_type("synthetic private output path")
        super().flush()


@pytest.mark.parametrize(("phase", "format", "comparison"), [
    ("write", "text", "pass"),
    ("flush", "text", "fail"),
    ("write", "json", "fail"),
    ("flush", "json", "pass"),
    ("write", "json", "invalid"),
])
def test_output_failure_uses_exit_two_and_sanitized_stderr(inputs, context, tmp_path, monkeypatch, phase, format, comparison):
    target = gate.BASELINE_PATH
    if comparison == "fail":
        target = write_candidate(tmp_path / "fail.json", failing_candidate(inputs[0], context))
    elif comparison == "invalid":
        target = tmp_path / "missing.json"
    stdout = FaultyOutput(phase, BrokenPipeError if format == "json" else OSError)
    stderr = StringIO()
    with monkeypatch.context() as patches:
        patches.setattr(sys, "stdout", stdout)
        patches.setattr(sys, "stderr", stderr)
        assert gate.main([str(target), "--format", format]) == 2
    assert stdout.flush_attempts == (phase == "flush")
    if phase == "write":
        assert stdout.write_attempts == 1 and not stdout.getvalue()
    assert stderr.getvalue() == "ERROR [output] Could not deliver checker output.\n"


@pytest.mark.parametrize("stderr_available", [False, True])
def test_unavailable_stderr_does_not_change_exit_two(monkeypatch, stderr_available):
    stdout = FaultyOutput("write")
    stderr = FaultyOutput("write") if stderr_available else None
    with monkeypatch.context() as patches:
        patches.setattr(sys, "stdout", stdout)
        patches.setattr(sys, "stderr", stderr)
        assert gate.main(["--format=json"]) == 2
    assert stdout.write_attempts == 1
    if stderr is not None:
        assert stderr.write_attempts == 1


@pytest.mark.parametrize(("format", "comparison"), [
    ("json", "pass"), ("text", "pass"), ("json", "fail"), ("json", "invalid"),
])
def test_subprocess_with_stdout_reader_closed_exits_two(inputs, context, tmp_path, format, comparison):
    target = gate.BASELINE_PATH
    if comparison == "fail":
        target = write_candidate(tmp_path / "fail.json", failing_candidate(inputs[0], context))
    elif comparison == "invalid":
        target = tmp_path / "missing.json"
    reader, writer = os.pipe()
    os.close(reader)
    try:
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", gate.__name__, str(target), "--format", format],
            cwd=gate.CORPUS_DIR.parents[1], stdout=writer, stderr=subprocess.PIPE,
        )
    finally:
        os.close(writer)
    with process:
        try:
            _, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    assert process.returncode == 2
    assert stderr.decode().splitlines() == ["ERROR [output] Could not deliver checker output."]
