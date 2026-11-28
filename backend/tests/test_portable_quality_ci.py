"""Exercise the CI adapter and wrapper steps used by portable-quality.yml."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest

from evaluation.portable_benchmark import check_quality_regression as gate
from evaluation.portable_benchmark import quality_ci_candidate as generator
from test_portable_quality_regression import failing_candidate


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/portable-quality.yml"
REPRODUCTION = gate.BASELINE_PATH.with_name("portable_quality_v1_run2.json")
ERROR_REPORT = {
    "schema_version": 1, "verdict": "ERROR", "validation": "invalid",
    "error": {"phase": "candidate", "message": "Invalid JSON or incompatible benchmark data."},
}
ADAPTER_STEP = "Verify report, publish summary and propagate status"
REF_STEP = "Reject unsupported benchmark refs"
TRAP_STEP = "Seed and generate a strict quality report"
REQUIRES_BASH = pytest.mark.skipif(
    os.name != "posix" or not shutil.which("bash"), reason="Hosted Bash step requires POSIX bash",
)


def workflow_step(name):
    # Only extract the run block of the named step. Changes to step wiring or
    # heredoc delimiters fail the tests instead of exercising a separate copy.
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.split(f"      - name: {name}\n")[1]
    step = start.split("\n      - name:", 1)[0]
    return dedent(step.split("        run: |\n", 1)[1]).strip() + "\n"


@pytest.fixture(scope="module")
def adapter_source():
    shell = workflow_step(ADAPTER_STEP)
    header, _, body = shell.partition("\n")
    assert header == "python - <<'PY'" and body.endswith("\nPY\n")
    return body.removesuffix("PY\n")


@pytest.fixture
def adapter(adapter_source):
    namespace = {"__name__": "workflow_test"}
    exec(compile(adapter_source, str(WORKFLOW), "exec"), namespace)
    return namespace["main"]


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    result = gate.run(REPRODUCTION)
    context = gate.trusted_context()
    baseline = gate.validate_report(gate.BASELINE_PATH.read_bytes(), context)
    path = tmp_path_factory.mktemp("ci-regression") / "report.json"
    path.write_text(failing_candidate(baseline, context).model_dump_json(), encoding="utf-8")
    return {0: (REPRODUCTION.read_bytes(), result), 1: (path.read_bytes(), gate.run(path)),
            2: (b"{", ERROR_REPORT)}


@pytest.fixture
def adapter_env(tmp_path):
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    workflow = WORKFLOW.read_text(encoding="utf-8")
    key_pattern = r"^  (TRUSTED_REFERENCE|[A-Z_]+SHA256): ([0-9a-f]+)$"
    return dict(
        re.findall(key_pattern, workflow, flags=re.MULTILINE),
        QUALITY_INPUT=str(inputs), QUALITY_OUTPUT=str(outputs), GITHUB_STEP_SUMMARY=str(tmp_path / "summary"),
        GITHUB_SHA="e" * 40, DOWNLOAD_OUTCOME="success", CANDIDATE_ARTIFACT="portable-quality-candidate-1.json",
    )


def candidate_path(env):
    return Path(env["QUALITY_INPUT"]) / env["CANDIDATE_ARTIFACT"]


def output_path(env, name):
    return Path(env["QUALITY_OUTPUT"]) / name


def execution(env):
    return json.loads(output_path(env, "execution.json").read_bytes())


def mock_checker(monkeypatch, code, diagnostics, expectation):
    if not isinstance(diagnostics, bytes):
        diagnostics = json.dumps(diagnostics).encode()
    fake = subprocess.CompletedProcess([], code, diagnostics, b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake)
    monkeypatch.setattr(gate, "run", lambda _: expectation)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_real_checker_status_propagation(adapter_source, adapter_env, reports, code):
    payload, _ = reports[code]
    candidate_path(adapter_env).write_bytes(payload)
    process = subprocess.run(
        [sys.executable, "-X", "utf8", "-"], input=adapter_source.encode(),
        env=dict(os.environ, **adapter_env), capture_output=True, timeout=180,
    )
    assert process.returncode == code, process.stderr.decode()
    record = execution(adapter_env)
    assert record["checker_exit_code"] == record["exit_code"] == code
    assert record["outcome"] == ("PASS", "QUALITY_REGRESSION", "INVALID_REPORT")[code]
    diagnostic = json.loads(output_path(adapter_env, "comparison.json").read_bytes())
    if code != 2:
        assert len(diagnostic["quality_checks"]) == 20 and len(diagnostic["categories"]) == 25
        assert record["checker_verdict"] == diagnostic["verdict"] == ("PASS", "FAIL")[code]
        assert "Nonblocking latency" in Path(adapter_env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    else:
        assert record["checker_verdict"] is None
        assert diagnostic["error"]["phase"] == "candidate" and diagnostic["error"]["message"]


@pytest.mark.parametrize("code", [0, 1, 2])
@pytest.mark.parametrize("unavailable", ["directory", "missing-variable", "missing-parent"])
def test_summary_failure_preserves_checker_result(adapter, adapter_env, reports, code, unavailable, monkeypatch, capsys):
    payload, diagnostic = reports[code]
    candidate_path(adapter_env).write_bytes(payload)
    mock_checker(monkeypatch, code, diagnostic, diagnostic)
    summary = Path(adapter_env["GITHUB_STEP_SUMMARY"])
    if unavailable == "directory":
        summary.mkdir()
    elif unavailable == "missing-variable":
        del adapter_env["GITHUB_STEP_SUMMARY"]
    else:
        adapter_env["GITHUB_STEP_SUMMARY"] = str(summary / "missing")
    assert adapter(adapter_env) == 2
    record = execution(adapter_env)
    assert record["outcome"] == "EXECUTION_ERROR" and record["adapter_status"] == "error"
    assert record["phase"] == "summary publication" and record["exit_code"] == 2
    assert record["checker_exit_code"] == code
    assert record["checker_verdict"] == ("PASS", "FAIL", None)[code]
    assert json.loads(output_path(adapter_env, "comparison.json").read_bytes()) == diagnostic
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr and str(summary) not in stderr
    assert "Adapter exit 2" in stderr


def test_original_summary_failure_subprocess(adapter_source, adapter_env, reports):
    candidate_path(adapter_env).write_bytes(reports[0][0])
    Path(adapter_env["GITHUB_STEP_SUMMARY"]).mkdir()
    process = subprocess.run(
        [sys.executable, "-X", "utf8", "-"], input=adapter_source.encode(),
        env=dict(os.environ, **adapter_env), capture_output=True, timeout=180,
    )
    assert process.returncode == 2 and b"Traceback" not in process.stderr
    record = execution(adapter_env)
    assert (record["outcome"], record["checker_verdict"], record["checker_exit_code"], record["exit_code"]) == (
        "EXECUTION_ERROR", "PASS", 0, 2,
    )


@pytest.mark.parametrize("filename", ["comparison.json", "execution.json"])
def test_durable_publication_failure(adapter, adapter_env, reports, filename, monkeypatch, capsys):
    payload, diagnostic = reports[0]
    candidate_path(adapter_env).write_bytes(payload)
    mock_checker(monkeypatch, 0, diagnostic, diagnostic)
    blocked = output_path(adapter_env, filename)
    blocked.mkdir()
    assert adapter(adapter_env) == 2
    assert blocked.is_dir() and list(blocked.iterdir()) == []
    assert {path.name for path in blocked.parent.iterdir()} == {"comparison.json", "execution.json"}
    if filename == "comparison.json":
        record = execution(adapter_env)
        assert record["outcome"] == "EXECUTION_ERROR" and record["checker_verdict"] == "PASS"
    else:
        assert json.loads(output_path(adapter_env, "comparison.json").read_bytes())["verdict"] == "PASS"
        assert "Validated checker quality verdict: PASS" in Path(
            adapter_env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    stderr = capsys.readouterr().err
    assert "Adapter exit 2" in stderr and "Traceback" not in stderr


@pytest.mark.parametrize(("mutation", "code"), [
    ("unexpected checker exit", 7), ("checker killed", -15), ("timeout", 0),
    ("no stdout", 0), ("whitespace", 0), ("malformed", 0), ("incomplete", 0),
    ("contradictory FAIL", 0), ("contradictory ERROR", 0), ("contradictory PASS", 1),
    ("original contradictory row", 0), ("drop", 0), ("tolerance", 0), ("baseline metric", 0),
    ("minimum", 0), ("quality range", 0), ("boolean numeric", 0), ("warning flag", 0),
    ("duplicate key", 0), ("missing error message", 2), ("policy identity", 0),
    ("candidate digest", 0), ("baseline digest", 0), ("duplicate check", 0),
    ("missing check", 0), ("unexpected category", 0), ("blocking count", 0), ("warning count", 0),
    ("ranking count", 0), ("latency delta", 0), ("nonfinite", 0),
])
def test_invalid_diagnostic_contract(adapter, adapter_env, reports, mutation, code, monkeypatch):
    payload, approved = reports[0]
    candidate_path(adapter_env).write_bytes(payload)
    diagnostic = deepcopy(approved)
    first = diagnostic["quality_checks"][0]
    top_level = {
        "contradictory FAIL": {"verdict": "FAIL"},
        "candidate digest": {"candidate_sha256": "0" * 64},
        "baseline digest": {"baseline_sha256": "0" * 64},
        "blocking count": {"blocking_failure_count": 1},
        "warning count": {"warning_count": 1},
        "ranking count": {"changed_ordered_rankings": 1},
    }
    fields = {
        "original contradictory row": {"candidate": 0.0, "drop": 1.0, "tolerance": 0.0},
        "drop": {"drop": 0.001}, "tolerance": {"tolerance": 0.5},
        "baseline metric": {"baseline": 0.3}, "minimum": {"minimum_acceptable": 0.0},
        "quality range": {"candidate": 1.1}, "boolean numeric": {"drop": False},
        "nonfinite": {"drop": float("inf")},
    }
    if mutation in top_level:
        diagnostic.update(top_level[mutation])
    elif mutation in fields:
        first.update(fields[mutation])
    elif mutation == "warning flag":
        diagnostic["categories"][0]["warning"] = True
    elif mutation == "policy identity":
        diagnostic["policy"]["sha256"] = "0" * 64
    elif mutation == "latency delta":
        diagnostic["latency"][0]["comparisons"][0]["change_ms"] = 1000000
    elif mutation == "duplicate check":
        diagnostic["quality_checks"][0] = diagnostic["quality_checks"][1]
    elif mutation == "missing check":
        diagnostic["quality_checks"].pop()
    elif mutation == "unexpected category":
        diagnostic["categories"][0]["category"] = "other"
    elif mutation == "contradictory ERROR":
        diagnostic = ERROR_REPORT
    elif mutation == "missing error message":
        diagnostic = {**ERROR_REPORT, "error": {"phase": "candidate"}}
    elif mutation in {"no stdout", "checker killed", "whitespace", "malformed", "incomplete", "duplicate key"}:
        diagnostic = {
            "no stdout": b"", "checker killed": b"", "whitespace": b" \n", "malformed": b"{",
            "incomplete": b'{"schema_version":1,"validation":"valid","verdict":"PASS"}',
            "duplicate key": b'{"schema_version":1,' + json.dumps(approved).encode()[1:],
        }[mutation]
    mock_checker(monkeypatch, code, diagnostic, approved)
    if mutation == "timeout":
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("checker", 120)

        monkeypatch.setattr(subprocess, "run", timeout)
    assert adapter(adapter_env) == 2
    record = execution(adapter_env)
    assert record["outcome"] == "EXECUTION_ERROR" and record["checker_verdict"] is None
    assert record["checker_exit_code"] == (None if mutation == "timeout" else code)
    assert not output_path(adapter_env, "comparison.json").exists()


@pytest.mark.parametrize("case", ["missing", "empty", "oversized", "extra", "directory", "download-failed", "bad-name"])
def test_input_layout_fails_before_checker(adapter, adapter_env, reports, case, monkeypatch):
    path = candidate_path(adapter_env)
    if case == "directory":
        path.mkdir()
    elif case != "missing":
        contents = {"empty": b"", "oversized": b" " * (1024 * 1024 + 1)}
        path.write_bytes(contents.get(case, reports[0][0]))
    if case == "extra":
        path.with_name("unexpected.txt").touch()
    if case == "download-failed":
        adapter_env["DOWNLOAD_OUTCOME"] = "failure"
    if case == "bad-name":
        adapter_env["CANDIDATE_ARTIFACT"] = "../untrusted.json"

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid artifact must not be given to the checker")

    monkeypatch.setattr(subprocess, "run", unexpected)
    assert adapter(adapter_env) == 2
    record = execution(adapter_env)
    assert record["phase"] == "candidate artifact transfer or layout" and record["checker_exit_code"] is None


def test_pass_with_latency_variance_and_epsilon_boundary(adapter, adapter_env, reports, monkeypatch):
    payload, expected = reports[0]
    candidate_path(adapter_env).write_bytes(payload)
    independently_recomputed = deepcopy(expected)
    row = independently_recomputed["quality_checks"][0]
    tolerance = row["tolerance"]
    baseline = tolerance + gate.EPSILON
    row.update(baseline=baseline, drop=baseline, candidate=0.0,
               minimum_acceptable=baseline - tolerance,
               decision="FAIL" if gate.exceeds_tolerance(baseline, tolerance, gate.EPSILON) else "PASS")
    assert row["decision"] == "PASS"
    assert any(stat["change_ms"] for group in expected["latency"] for stat in group["comparisons"])
    mock_checker(monkeypatch, 0, independently_recomputed, independently_recomputed)
    assert adapter(adapter_env) == 0
    contradictory = deepcopy(independently_recomputed)
    contradictory["quality_checks"][0]["decision"] = "FAIL"
    mock_checker(monkeypatch, 0, contradictory, independently_recomputed)
    assert adapter(adapter_env) == 2


@pytest.mark.parametrize(("state", "before", "after", "stop_error", "failure"), [
    (None, None, None, None, None),
    ("ready", {"network": ["old"]}, {"network": ["old"]}, None, None),
    ("starting", {}, {}, None, None),
    ("stopped", {}, {}, None, None),
    (None, {}, {}, None, None),
    (None, {}, {"network": ["extra"]}, None, generator.InventoryEvidenceError),
    ("invalid", {}, None, None, ValueError),
    ("ready", {}, {}, OSError("stop failed"), OSError),
    ("ready", {}, {}, ValueError("replacement identity"), ValueError),
    ("stopped", {}, OSError("inventory failed"), None, OSError),
    ("stopped", {}, {"network": ["extra"]}, None, generator.InventoryEvidenceError),
    ("ready", None, {}, None, generator.InventoryEvidenceError),
    ("stopped", None, {}, None, generator.InventoryEvidenceError),
])
def test_cleanup_evidence(tmp_path, state, before, after, stop_error, failure, monkeypatch, capsys):
    calls = []
    path = tmp_path / "ownership.json"
    if state is not None:
        path.touch()
    if before is not None:
        (tmp_path / "inventory.json").write_text(json.dumps(before), encoding="utf-8")

    def load_state(_):
        if state == "invalid":
            raise ValueError("invalid ownership")
        return SimpleNamespace(phase=state)

    def stop(*args):
        calls.append(args)
        if stop_error:
            raise stop_error

    def inventory():
        if isinstance(after, Exception):
            raise after
        return after

    monkeypatch.setattr(generator, "benchmark", stop)
    monkeypatch.setattr(generator, "load_state", load_state)
    monkeypatch.setattr(generator, "inventory", inventory)
    if failure:
        with pytest.raises(failure):
            list(generator.cleanup(tmp_path))
        assert "inventory check completed" not in capsys.readouterr().out
    else:
        list(generator.cleanup(tmp_path))
    assert calls == ([("seed_lifecycle", "stop", "--state", path)] if state in {"ready", "starting"} else [])


def test_repeated_cleanup_keeps_missing_inventory_failure(tmp_path, monkeypatch):
    (tmp_path / "ownership.json").touch()
    (tmp_path / "startup.attempted").touch()
    state = SimpleNamespace(phase="ready")
    monkeypatch.setattr(generator, "load_state", lambda _: state)
    monkeypatch.setattr(generator, "benchmark", lambda *args: setattr(state, "phase", "stopped"))
    for expected_phase in ("stopped", "stopped"):
        with pytest.raises(generator.InventoryEvidenceError, match="unavailable"):
            list(generator.cleanup(tmp_path))
        assert state.phase == expected_phase


@pytest.mark.parametrize("with_inventory", [False, True])
def test_missing_ownership_after_attempt_fails(tmp_path, with_inventory, monkeypatch):
    (tmp_path / "startup.attempted").touch()
    if with_inventory:
        (tmp_path / "inventory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(generator, "inventory", lambda: {})
    monkeypatch.setattr(generator, "benchmark", lambda *args: pytest.fail("No owned resources to stop"))
    reason = "no ownership record" if with_inventory else "unavailable"
    with pytest.raises(generator.InventoryEvidenceError, match=reason):
        list(generator.cleanup(tmp_path))


@pytest.mark.parametrize("failure", [
    None, "preflight", "initial inventory", "start", "readiness", "seed", "runner", "no report", "validation",
])
def test_generation_failures_do_not_fall_back(tmp_path, failure, monkeypatch):
    report = tmp_path / "candidate.json"
    called = []
    ids = {str(index) for index in range(40)}
    docs = [
        SimpleNamespace(passages=[
            SimpleNamespace(passage_id=str(doc * 5 + index), vector_id=str(index)) for index in range(5)
        ]) for doc in range(8)
    ]

    def fail_at(stage):
        if failure == stage:
            raise ValueError(stage)

    def benchmark(module, *args):
        called.append(module)
        fail_at({"seed_lifecycle": "start", "seed": "seed", "run_benchmark": "runner"}[module])
        if module == "run_benchmark" and failure != "no report":
            report.write_bytes(b"{}")
        return b"{}"

    def validate(*args):
        fail_at("validation")
        return SimpleNamespace(results=[SimpleNamespace(queries=range(36)) for _ in range(5)])

    monkeypatch.setenv("AURA_TEST_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("AURA_QUALITY_RERANKER", "fake")
    monkeypatch.setattr(generator, "preflight", lambda _: fail_at("preflight"))
    monkeypatch.setattr(generator, "inventory", lambda: fail_at("initial inventory") or {})
    monkeypatch.setattr(generator, "benchmark", benchmark)
    monkeypatch.setattr(generator, "load_state", lambda _: SimpleNamespace(
        phase="starting" if failure == "readiness" else "ready", context="default", project="fake",
    ))
    monkeypatch.setattr(generator, "SeedRecord", SimpleNamespace(model_validate_json=lambda _: SimpleNamespace(
        documents=docs, alembic_revision="625afa44aa81", vector_dimensions=384, corpus_fingerprint="sample",
    )))
    monkeypatch.setattr(generator, "trusted_context", lambda: SimpleNamespace(
        passage_ids=ids, manifest=SimpleNamespace(corpus_fingerprint="sample"),
    ))
    monkeypatch.setattr(generator, "validate_report", validate)
    expected_calls = {
        "preflight": [], "initial inventory": [], "start": ["seed_lifecycle"], "readiness": ["seed_lifecycle"],
        "seed": ["seed_lifecycle", "seed"],
    }
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            list(generator.generate(tmp_path, report))
    else:
        list(generator.generate(tmp_path, report))
    assert called == expected_calls.get(failure, ["seed_lifecycle", "seed", "run_benchmark"])


def test_ref_guard_precedes_setup_and_verify_depends_on_generation():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    generate = workflow.split("  generate:\n")[1].split("\n  verify:")[0]
    assert "    if:" not in generate.split("    steps:")[0]
    assert generate.split("    steps:\n")[1].startswith(
        f"      - name: {REF_STEP}\n        working-directory: .\n"
    )
    assert "needs: generate" in workflow.split("\n  verify:")[1]
    assert "branches: [main]" in workflow and "pull_request" not in workflow


@REQUIRES_BASH
@pytest.mark.parametrize(("ref", "status"), [
    ("refs/heads/main", 0), ("refs/heads/feature", 2), ("refs/tags/main", 2),
])
def test_actual_ref_guard(ref, status):
    result = subprocess.run(["bash", "-e", "-c", workflow_step(REF_STEP)],
                            env=dict(os.environ, GITHUB_REF=ref), capture_output=True, timeout=10)
    assert result.returncode == status
    assert (b"requires main" in result.stderr) == bool(status)


@REQUIRES_BASH
@pytest.mark.parametrize(("generation", "cleanup", "status"), [(0, 0, 0), (0, 2, 2), (3, 2, 3)])
def test_actual_generation_trap(tmp_path, generation, cleanup, status):
    env = dict(os.environ, QUALITY_WORK=str(tmp_path), GENERATION=str(generation), CLEANUP=str(cleanup))
    functions = """
python() {
  if [[ "$3" == generate ]]; then
    echo 'generation evidence'
    return "$GENERATION"
  fi
  echo 'cleanup evidence'
  return "$CLEANUP"
}
timeout() { shift; "$@"; }
"""
    shell = functions + workflow_step(TRAP_STEP)
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", shell],
                            env=env, capture_output=True, timeout=10)
    assert result.returncode == status
    assert (tmp_path / "generation.txt").read_text().strip() == "generation evidence"
    assert (tmp_path / "cleanup.txt").read_text().splitlines() == [
        "cleanup evidence", f"Generation exit {generation}; cleanup trap exit {cleanup}.",
    ]
