import argparse
import json
from pathlib import Path

from evaluation.metrics.retrieval_metrics import RANKING_METRICS

BASELINE_PATH = (
    Path(__file__).parent / "results" / "retrieval_benchmark_results.json"
)
METADATA_FIELDS = (
    "dataset_name", "dataset_version", "workspace_id", "top_k", "query_count",
)
ROUNDING_TOLERANCE = 1e-12


def unit_interval(value):
    try:
        score = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1") from error
    if not 0 <= score <= 1:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return score


def metric_minimum(value):
    metric, _, score = value.partition("=")
    if metric not in RANKING_METRICS:
        raise argparse.ArgumentTypeError(f"expected METRIC=VALUE for one of {RANKING_METRICS}")
    return metric, unit_interval(score)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_results(path):
    try:
        with path.open(encoding="utf-8") as file:
            report = json.load(file, object_pairs_hook=reject_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return report


def index_by(rows, field, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label}: expected a list of rows indexed by {field}")

    indexed = {}
    for row in rows:
        key = row.get(field) if isinstance(row, dict) else None
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f"{label}: invalid or duplicate {field}: {key!r}")
        indexed[key] = row
    return indexed


def require_matching_keys(expected, actual, label):
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"{label} differ (missing: {missing}, unexpected: {extra})")


def validate_compatibility(baseline, current):
    for field in METADATA_FIELDS:
        before = baseline[field]
        after = current[field]
        if field in ("dataset_name", "dataset_version"):
            valid = all(isinstance(value, str) and value.strip() for value in (before, after))
        else:
            valid = all(type(value) is int and value > 0 for value in (before, after))
        if not valid:
            raise ValueError(f"invalid benchmark metadata for {field}: {before!r}, {after!r}")
        if before != after:
            raise ValueError(f"{field} differs: baseline {before!r}, current {after!r}")

    baseline_configs = index_by(baseline["retrieval_configs"], "name", "baseline configurations")
    current_configs = index_by(current["retrieval_configs"], "name", "current configurations")
    if not baseline_configs:
        raise ValueError("baseline retrieval configurations are empty")
    require_matching_keys(baseline_configs, current_configs, "retrieval configuration names")

    for name, config in baseline_configs.items():
        current_config = current_configs[name]
        for label, entry in (("baseline", config), ("current", current_config)):
            mode = entry["mode"]
            pool_size = entry["candidate_pool_size"]
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError(f"{label} / {name}: invalid retrieval mode {mode!r}")
            if pool_size is not None and (type(pool_size) is not int or pool_size < 1):
                raise ValueError(f"{label} / {name}: invalid candidate pool size {pool_size!r}")
        if config != current_config:
            raise ValueError(
                f"{name}: retrieval settings differ "
                f"(baseline {config!r}, current {current_config!r})"
            )

    for label, report in (("baseline", baseline), ("current", current)):
        results = report["results"]
        if not isinstance(results, dict):
            raise ValueError(f"{label} results must be a JSON object")
        require_matching_keys(baseline_configs, results, f"{label} result configurations")
        for name, config in baseline_configs.items():
            if not isinstance(results[name], dict):
                raise ValueError(f"{label} / {name}: expected a result object")
            for field in ("mode", "candidate_pool_size"):
                actual = results[name][field]
                expected = config[field]
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError(
                        f"{label} / {name}: result {field} is {actual!r}, expected {expected!r}"
                    )


def get_score(row, metric, label):
    score = row.get(metric)
    if type(score) not in (float, int) or not 0 <= score <= 1:
        raise ValueError(f"{label}: {metric} must be a number between 0 and 1")
    return score


def compare_scores(baseline, current, label, tolerance, minimums):
    for metric in RANKING_METRICS:
        before = get_score(baseline, metric, label)
        after = get_score(current, metric, label)
        if before - after > tolerance + ROUNDING_TOLERANCE:
            yield (
                f"{label} / {metric}: {before:.6f} -> {after:.6f} "
                f"(drop {before - after:.6f}, allowed {tolerance:.6f})"
            )
        if metric in minimums and minimums[metric] - after > ROUNDING_TOLERANCE:
            yield (
                f"{label} / {metric}: {after:.6f} "
                f"is below the minimum {minimums[metric]:.6f}"
            )


def compare_results(baseline, current, aggregate_tolerance, query_tolerance, minimums):
    validate_compatibility(baseline, current)
    failures = []
    expected_queries = None

    for name, before in baseline["results"].items():
        after = current["results"][name]
        before_queries = index_by(before["per_query"], "query_id", f"baseline / {name}")
        after_queries = index_by(after["per_query"], "query_id", f"current / {name}")
        require_matching_keys(before_queries, after_queries, f"{name} query IDs")
        if len(before_queries) != baseline["query_count"]:
            raise ValueError(
                f"{name}: expected {baseline['query_count']} query results, got {len(before_queries)}"
            )
        if expected_queries is None:
            expected_queries = before_queries
        require_matching_keys(
            expected_queries, before_queries, f"{name} query IDs across configurations"
        )

        failures.extend(compare_scores(before, after, name, aggregate_tolerance, minimums))
        for query_id, before_query in before_queries.items():
            after_query = after_queries[query_id]
            label = f"{name} / {query_id}"
            category = before_query["category"]
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"{label}: invalid baseline query category {category!r}")
            if category != after_query["category"]:
                raise ValueError(
                    f"{label}: query category differs "
                    f"(baseline {category!r}, current {after_query['category']!r})"
                )
            failures.extend(
                compare_scores(before_query, after_query, label, query_tolerance, {})
            )

    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Check retrieval benchmark results for quality regressions."
    )
    parser.add_argument("current", type=Path, help="Fresh benchmark output")
    parser.add_argument(
        "--baseline", type=Path, default=BASELINE_PATH, help="Reference benchmark output"
    )
    parser.add_argument(
        "--aggregate-tolerance", type=unit_interval, default=0.0,
        help="Allowed aggregate score drop",
    )
    parser.add_argument(
        "--query-tolerance", type=unit_interval, default=0.0,
        help="Allowed per-query score drop",
    )
    parser.add_argument(
        "--minimum", type=metric_minimum, action="append", default=[],
        metavar="METRIC=VALUE",
        help="Minimum aggregate score across all configurations (repeatable)",
    )
    args = parser.parse_args()
    minimums = dict(args.minimum)
    if len(minimums) != len(args.minimum):
        parser.error("specify each minimum at most once")

    try:
        if args.baseline.samefile(args.current):
            raise ValueError("baseline and current results must be separate files")
        failures = compare_results(
            load_results(args.baseline), load_results(args.current),
            args.aggregate_tolerance, args.query_tolerance, minimums,
        )
    except KeyError as error:
        parser.error(f"benchmark report is missing required field {error}")
    except (OSError, ValueError) as error:
        parser.error(f"cannot compare benchmark results: {error}")

    if failures:
        print(f"Retrieval quality check failed ({len(failures)} findings):")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("Retrieval quality check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
