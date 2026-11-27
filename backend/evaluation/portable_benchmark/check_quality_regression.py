"""Compare portable quality reports against the reviewed v1 baseline."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path
import statistics
import sys
from typing import Literal

from pydantic import ValidationError, model_validator

from evaluation.portable_benchmark.corpus import CORPUS_DIR, JSONNumberUnderflow, decode_json, json_digest
from evaluation.portable_benchmark.ranking_report import Quality, UnitFloat, mean_quality, quality_scores, validate_report
from evaluation.portable_benchmark.schema import Digest, StrictModel, VersionOne


BASELINE_PATH = CORPUS_DIR / "baselines" / "portable_quality_v1.json"
POLICY_PATH = CORPUS_DIR / "baselines" / "portable_quality_v1_policy.json"
BASELINE_SHA256 = "65d8b820a5accccb28462788bd2465d868f105ff52bcadf3036527c0e89dea4e"
DATASET_SHA256 = "bf6cb3182ff43ae9b447e9d4ff37086d0a67d1d7f9edcc39132f29f79e7f38d6"
CORPUS_SHA256 = "e4f0f2fba8a9c2b598a260269891cde0b54d9566b0c0a68beb74443a7d700300"
EPSILON = 1e-12
METRICS = tuple(Quality.model_fields)


class Policy(StrictModel):
    schema_version: VersionOne
    policy_id: Literal["portable-quality-v1"]
    baseline_sha256: Digest
    blocking_tolerances: Quality
    category_ndcg_warning: UnitFloat
    epsilon: UnitFloat

    @model_validator(mode="after")
    def pin_reference_and_epsilon(self):
        if self.baseline_sha256 != BASELINE_SHA256:
            raise ValueError("Policy must identify the authoritative v1 baseline")
        if self.epsilon != EPSILON:
            raise ValueError("Policy epsilon must equal 1e-12")
        return self


class GateError(ValueError):
    def __init__(self, phase, message):
        super().__init__(message)
        self.phase = phase


def approved_baseline_bytes(raw):
    if sha256(raw).hexdigest() == BASELINE_SHA256:
        return raw
    normalized = raw.replace(b"\r\n", b"\n")
    if normalized != raw and sha256(normalized).hexdigest() == BASELINE_SHA256:
        return normalized
    raise GateError("baseline", "Baseline bytes do not match the approved SHA-256.")


def trusted_context():
    # The shared loader validates files and manifests without loading models or contacting services.
    from evaluation.portable_benchmark.run_benchmark import load_report_context

    context = load_report_context(
        "quality", top_k=5, hybrid_pool=20, rerank_pools=(5, 10, 20),
        dataset_path=CORPUS_DIR / "dataset.reviewed.json", expected_dataset_sha256=DATASET_SHA256,
    )
    if (
        context.manifest.corpus_fingerprint != CORPUS_SHA256
        or len(context.manifest.documents) != 8
        or context.dataset.schema_version != 2 or len(context.dataset.queries) != 36
        or len(context.passage_ids) != 40
        or any({j.passage_id for j in q.judgments} != context.passage_ids for q in context.dataset.queries)
    ):
        raise GateError("context", "Trusted inputs do not match the approved corpus and judgment inventory.")
    return context


def exceeds_tolerance(drop, tolerance, epsilon):
    return drop > tolerance + epsilon


def metric_delta(metric, baseline, candidate):
    return {"metric": metric, "baseline": baseline, "candidate": candidate, "drop": baseline - candidate}


def ranking_delta(profile, before, after):
    old = {item.passage_id: (rank, item.score) for rank, item in enumerate(before.ranking, 1)}
    new = {item.passage_id: (rank, item.score) for rank, item in enumerate(after.ranking, 1)}
    return {
        "profile": profile, "query_id": before.query_id,
        "ordered_ranking_changed": list(old) != list(new),
        "added": [{"passage_id": pid, "rank": rank, "score": score}
                  for pid, (rank, score) in new.items() if pid not in old],
        "removed": [{"passage_id": pid, "rank": rank, "score": score}
                    for pid, (rank, score) in old.items() if pid not in new],
        "moved": [{"passage_id": pid, "baseline_rank": old[pid][0], "candidate_rank": rank}
                  for pid, (rank, _) in new.items() if pid in old and old[pid][0] != rank],
        "score_changes": [{"passage_id": pid, "baseline_score": old[pid][1], "candidate_score": score}
                          for pid, (_, score) in new.items() if pid in old and old[pid][1] != score],
    }


def latency_summary(queries):
    values = sorted(query.latency_ms for query in queries)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else statistics.mean(values[middle - 1:middle + 1])
    lower, remainder = divmod((len(values) - 1) * 95, 100)
    # Inclusive p95, using interpolation that also handles very large finite observations.
    p95 = values[lower]
    if remainder:
        p95 += (values[lower + 1] - p95) * (remainder / 100)
    return {"mean_ms": statistics.mean(values), "median_ms": median, "p95_ms": p95}


def latency_delta(profile, before, after):
    old, new = latency_summary(before.queries), latency_summary(after.queries)
    comparisons = []
    for metric, baseline in old.items():
        candidate = new[metric]
        change = candidate - baseline
        percent = change / baseline * 100 if baseline else None
        comparisons.append({
            "metric": metric, "baseline": baseline, "candidate": candidate, "change_ms": change,
            "change_percent": percent if percent is not None and isfinite(percent) else None,
        })
    return {
        "profile": profile, "comparisons": comparisons,
        "inconsistent_summaries": [
            {"report": label, "metric": metric}
            for label, actual, recomputed in (
                ("baseline", before.latency, old), ("candidate", after.latency, new),
            )
            for metric, value in recomputed.items()
            if not isclose(value, getattr(actual, metric), rel_tol=1e-12, abs_tol=1e-9)
        ],
    }


def compare_reports(baseline, candidate, context, policy):
    """Compare reports already validated against the supplied trusted context."""
    checks, categories, query_changes, rankings, latency = [], [], [], [], []
    for before, after in zip(baseline.results, candidate.results, strict=True):
        profile = before.configuration.name
        old_categories, new_categories = defaultdict(list), defaultdict(list)
        old_scores, new_scores = [], []
        for old, new, query in zip(before.queries, after.queries, context.dataset.queries, strict=True):
            old_quality = quality_scores([row.passage_id for row in old.ranking], query.judgments, context.top_k)
            new_quality = quality_scores([row.passage_id for row in new.ranking], query.judgments, context.top_k)
            old_scores.append(old_quality)
            new_scores.append(new_quality)
            old_categories[query.category].append(old_quality)
            new_categories[query.category].append(new_quality)

            changes = [
                metric_delta(metric, getattr(old_quality, metric), getattr(new_quality, metric))
                for metric in METRICS
                if abs(getattr(old_quality, metric) - getattr(new_quality, metric)) > policy.epsilon
            ]
            if changes:
                query_changes.append({
                    "profile": profile, "query_id": query.query_id, "category": query.category,
                    "metrics": changes,
                })
            if old.ranking != new.ranking:
                rankings.append(ranking_delta(profile, old, new))

        old_mean, new_mean = mean_quality(old_scores), mean_quality(new_scores)
        for metric in METRICS:
            tolerance = getattr(policy.blocking_tolerances, metric)
            delta = metric_delta(metric, old_mean[metric], new_mean[metric])
            checks.append({
                "profile": profile, **delta, "tolerance": tolerance,
                "minimum_acceptable": max(0.0, delta["baseline"] - tolerance),
                "decision": "FAIL" if exceeds_tolerance(delta["drop"], tolerance, policy.epsilon) else "PASS",
            })
        for category, rows in sorted(old_categories.items()):
            delta = metric_delta(
                "ndcg_at_k", statistics.mean(q.ndcg_at_k for q in rows),
                statistics.mean(q.ndcg_at_k for q in new_categories[category]),
            )
            categories.append({
                "profile": profile, "category": category, "query_count": len(rows), **delta,
                "warning_threshold": policy.category_ndcg_warning,
                "warning": exceeds_tolerance(delta["drop"], policy.category_ndcg_warning, policy.epsilon),
            })
        latency.append(latency_delta(profile, before, after))

    failure_count = sum(check["decision"] == "FAIL" for check in checks)
    category_count = sum(row["warning"] for row in categories)
    latency_count = sum(len(row["inconsistent_summaries"]) for row in latency)
    query_changes.sort(key=lambda row: (-max(0, *(m["drop"] for m in row["metrics"])), row["profile"], row["query_id"]))
    return {
        "verdict": "FAIL" if failure_count else "PASS",
        "blocking_failure_count": failure_count, "warning_count": category_count + latency_count,
        "quality_checks": checks, "categories": categories, "query_changes": query_changes,
        "changed_ordered_rankings": sum(row["ordered_ranking_changed"] for row in rankings),
        "ranking_changes": rankings, "latency": latency,
    }


def validate_input(phase, operation):
    try:
        return operation()
    except GateError:
        raise
    except OSError:
        raise GateError(phase, "Could not read the required input.") from None
    except JSONNumberUnderflow:
        raise GateError(phase, "Nonzero JSON number underflows to zero.") from None
    except ValidationError as error:
        # The schema's error type is safe. Locations and input values may be user-controlled.
        kind = error.errors(include_url=False, include_context=False, include_input=False)[0]["type"]
        raise GateError(phase, f"Strict validation rejected the input ({kind}).") from None
    except ValueError as error:
        detail = str(error)
        if ": duplicate JSON key " in detail:
            reason = "Duplicate JSON key."
        elif ": nonfinite JSON number " in detail:
            reason = "Nonfinite JSON number."
        else:
            reason = "Invalid JSON or incompatible benchmark data."
        raise GateError(phase, reason) from None
    except Exception as error:
        raise GateError(phase, f"Could not complete {phase} ({type(error).__name__}).") from None


def run(candidate_path):
    policy = validate_input("policy", lambda: Policy.model_validate(decode_json(POLICY_PATH.read_bytes(), "Policy")))
    raw = validate_input("baseline", BASELINE_PATH.read_bytes)
    baseline_bytes = approved_baseline_bytes(raw)
    context = validate_input("context", trusted_context)
    baseline = validate_input("baseline", lambda: validate_report(baseline_bytes, context))
    candidate_bytes = validate_input("candidate", candidate_path.read_bytes)
    candidate = validate_input("candidate", lambda: validate_report(candidate_bytes, context))
    result = validate_input("comparison", lambda: compare_reports(baseline, candidate, context, policy))
    return {
        "schema_version": 1, "validation": "valid",
        "baseline_sha256": sha256(baseline_bytes).hexdigest(), "baseline_input_sha256": sha256(raw).hexdigest(),
        "candidate_sha256": sha256(candidate_bytes).hexdigest(),
        "policy": {"id": policy.policy_id, "version": policy.schema_version, "sha256": json_digest(policy.model_dump()),
                   "epsilon": policy.epsilon},
        "context": {"dataset_fingerprint": baseline.dataset_fingerprint, "corpus_fingerprint": baseline.corpus_fingerprint,
                    "evaluation_protocol": baseline.evaluation_protocol, "top_k": context.top_k},
        **result,
    }


def print_text(result):
    print(f"Validated both reports. Baseline {result['baseline_sha256']}.")
    print(f"Policy {result['policy']['id']} v{result['policy']['version']} ({result['policy']['sha256']}).")
    for row in result["quality_checks"]:
        print(
            f"{row['decision']} {row['profile']} {row['metric']}: "
            f"{row['baseline']:.6f} -> {row['candidate']:.6f} "
            f"(drop {row['drop']:+.6f}, allowed {row['tolerance']:.6f}, minimum {row['minimum_acceptable']:.6f})"
        )
    for row in result["categories"]:
        if row["warning"]:
            print(f"WARNING {row['profile']} / {row['category']} nDCG drop {row['drop']:.6f} ({row['query_count']} queries)")
    for row in [r for r in result["query_changes"] if any(m["drop"] > 0 for m in r["metrics"])][:5]:
        changes = ", ".join(f"{m['metric']} {m['drop']:+.4f}" for m in row["metrics"])
        print(f"Query {row['profile']} / {row['query_id']} drops: {changes}")
    print(f"Changed ordered rankings: {result['changed_ordered_rankings']}; ranking or score changes: {len(result['ranking_changes'])}")
    for row in result["latency"]:
        for metric in row["comparisons"]:
            percent = metric["change_percent"]
            percent_text = "n/a" if percent is None else f"{percent:+.2f}%"
            print(
                f"Latency {row['profile']} {metric['metric']}: {metric['baseline']:.2f} -> "
                f"{metric['candidate']:.2f} ms ({metric['change_ms']:+.2f} ms, {percent_text})"
            )
        for warning in row["inconsistent_summaries"]:
            print(f"WARNING {row['profile']} {warning['report']} {warning['metric']} disagrees with observations")
    print(f"{result['verdict']}: {result['blocking_failure_count']} blocking failures, {result['warning_count']} warnings.")


def discard_failed_buffer(stream):
    if stream is None or stream.closed:
        return
    try:
        # Python can retry a failed flush at shutdown and override the requested exit code.
        # Keep the descriptor usable while discarding any remaining buffered bytes.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), stream.fileno())
    except (OSError, ValueError):
        # An already closed descriptor or in-memory stream needs no redirection.
        pass


def deliver_output(stream, write, status):
    try:
        if stream is None or stream.closed:
            raise OSError("Output stream unavailable")
        write()
        stream.flush()
    except OSError:
        discard_failed_buffer(stream)
        stderr = sys.stderr
        if stderr is not stream and stderr is not None and not stderr.closed:
            try:
                print("ERROR [output] Could not deliver checker output.", file=stderr, flush=True)
            except OSError:
                discard_failed_buffer(stderr)
        return 2
    return status


class ArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise GateError("arguments", "Supply a candidate path and optionally --format text or --format json.")

    def print_help(self, file=None):
        stream = sys.stdout if file is None else file
        if deliver_output(stream, lambda: stream.write(self.format_help()), 0):
            self.exit(2)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_output = "--format=json" in argv or any(
        (flag, value) == ("--format", "json") for flag, value in zip(argv, argv[1:])
    )
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("candidate", type=Path, help="Portable quality report to evaluate")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    try:
        args = parser.parse_args(argv)
        json_output = args.format == "json"
        with redirect_stdout(sys.stderr):
            result = run(args.candidate)
    except GateError as error:
        if json_output:
            result = {"schema_version": 1, "verdict": "ERROR", "validation": "invalid",
                      "error": {"phase": error.phase, "message": str(error)}}
            return deliver_output(sys.stdout, lambda: print(json.dumps(result, allow_nan=False)), 2)
        return deliver_output(sys.stderr, lambda: print(f"ERROR [{error.phase}] {error}", file=sys.stderr), 2)
    status = 1 if result["blocking_failure_count"] else 0
    if json_output:
        return deliver_output(sys.stdout, lambda: print(json.dumps(result, allow_nan=False)), status)
    return deliver_output(sys.stdout, lambda: print_text(result), status)


if __name__ == "__main__":
    sys.exit(main())
