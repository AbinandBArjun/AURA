"""Generate a portable quality CI candidate and clean up its owned Docker resources."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from evaluation.portable_benchmark.check_quality_regression import trusted_context
from evaluation.portable_benchmark.ranking_report import validate_report
from evaluation.portable_benchmark.report_output import output_destination, posix_destination
from evaluation.portable_benchmark.seed_lifecycle import load_state, verify_ports
from evaluation.portable_benchmark.seed_record import SeedRecord


PACKAGE = "evaluation.portable_benchmark."
CONTEXT = "default"
PORTS = ("--postgres-port", "15432", "--qdrant-port", "16333")
DATASET = "evaluation/portable_benchmark/dataset.reviewed.json"
DATASET_SHA256 = "bf6cb3182ff43ae9b447e9d4ff37086d0a67d1d7f9edcc39132f29f79e7f38d6"


class InventoryEvidenceError(ValueError):
    pass


def run(*command):
    return subprocess.run(command, check=True, capture_output=True).stdout


def benchmark(module, *args):
    return run(sys.executable, "-m", PACKAGE + module, *args)


def inventory():
    # Keep full identities private. Include stopped containers and named volumes.
    return {
        kind: sorted(run("docker", "--context", CONTEXT, kind, "ls", *flags,
                         "--format", fields).decode().splitlines())
        for kind, flags, fields in (
            ("container", ("-a", "--no-trunc"), "{{.ID}} {{.Names}} {{.State}}"),
            ("network", ("--no-trunc",), "{{.ID}} {{.Name}}"),
            ("volume", (), "{{.Name}} {{.Driver}}"),
        )
    }


def preflight(output_dir):
    verify_ports(15432, 16333)
    with TemporaryDirectory(dir=output_dir) as probe:
        path = Path(probe) / "probe.json"
        payload = b'{"publication_probe":true}\n'
        with posix_destination(output_destination(path)) as (stream, commit):
            stream.write(payload)
            stream.flush()
            commit()
        if path.read_bytes() != payload:
            raise ValueError("Publication probe differs from its staged payload")


def generate(root, report):
    yield "loopback ports and Linux atomic report publication"
    preflight(report.parent)
    yield "pre-run Docker inventory"
    (root / "inventory.json").write_text(json.dumps(inventory()), encoding="utf-8")
    yield "owned infrastructure startup"
    state_path = root / "ownership.json"
    (root / "startup.attempted").touch(exist_ok=False)
    benchmark("seed_lifecycle", "start", "--state", state_path, "--docker-context", CONTEXT, *PORTS)
    state = load_state(state_path)
    if state.phase != "ready":
        raise ValueError("Benchmark stack is not ready")
    target = ("--docker-context", state.context, "--project", state.project, *PORTS)

    yield "seed and verify migrations, corpus and vectors"
    embedding = os.environ["AURA_TEST_EMBEDDING_MODEL"]
    seed = SeedRecord.model_validate_json(benchmark("seed", *target, "--model-dir", embedding))
    context = trusted_context()
    passages = [passage for document in seed.documents for passage in document.passages]
    if (
        not re.fullmatch(r"[0-9a-f]{12}", seed.alembic_revision) or seed.vector_dimensions != 384
        or seed.corpus_fingerprint != context.manifest.corpus_fingerprint
        or len(seed.documents) != 8 or len(passages) != 40
        or {p.passage_id for p in passages} != context.passage_ids
        or any(p.vector_id is None for p in passages)
    ):
        raise ValueError("Seed does not match the approved inventory")

    print("Verified eight documents, 40 canonical chunks and 40 matching 384-dimensional cosine points.", flush=True)
    print(f"Verified checked-in Alembic head {seed.alembic_revision}.", flush=True)

    yield "generate candidate report"
    benchmark(
        "run_benchmark", *target, "--embedding-model-dir", embedding,
        "--reranker-model-dir", os.environ["AURA_QUALITY_RERANKER"],
        "--mode", "quality", "--dataset", DATASET, "--dataset-sha256", DATASET_SHA256,
        "--top-k", "5", "--hybrid-pool", "20", "--rerank-pools", "5", "10", "20",
        "--output", report,
    )
    yield "strict candidate validation"
    if not 0 < report.stat().st_size <= 1024 * 1024:
        raise ValueError("Candidate report must fit the CI size bound")
    raw = report.read_bytes()
    validated = validate_report(raw, context)
    if sum(len(profile.queries) for profile in validated.results) != 180:
        raise ValueError("Unexpected candidate result count")
    print(f"Validated 180 results. Candidate report SHA-256 {sha256(raw).hexdigest()}", flush=True)


def cleanup(root):
    yield "ownership-checked teardown"
    state_path = root / "ownership.json"
    has_ownership = state_path.exists()
    if has_ownership and load_state(state_path).phase != "stopped":
        benchmark("seed_lifecycle", "stop", "--state", state_path)

    yield "post-run Docker inventory"
    before = root / "inventory.json"
    attempted = (root / "startup.attempted").exists()
    if not before.exists():
        if has_ownership or attempted:
            raise InventoryEvidenceError("Pre-run Docker inventory is unavailable after attempted startup")
        print("Infrastructure startup had not begun. No inventory comparison was needed.", flush=True)
        return
    if inventory() != json.loads(before.read_bytes()):
        raise InventoryEvidenceError("Docker inventory changed after the benchmark")
    if has_ownership:
        print("Owned-resource cleanup and inventory check completed.", flush=True)
    elif attempted:
        raise InventoryEvidenceError("Docker inventory matches, but attempted startup has no ownership record")
    else:
        print("Startup had not begun. Docker inventory matches the pre-run snapshot.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "cleanup"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    phase = args.operation
    try:
        if args.operation == "generate":
            attempt = os.environ["GITHUB_RUN_ATTEMPT"]
            if not re.fullmatch(r"[1-9][0-9]*", attempt):
                raise ValueError("Invalid workflow run attempt")
            report = args.directory / f"portable-quality-candidate-{attempt}.json"
            steps = generate(args.directory, report)
        else:
            steps = cleanup(args.directory)
        for phase in steps:
            print(f"Portable quality CI | {phase}", flush=True)
    except InventoryEvidenceError as error:
        print(f"Cleanup inventory verification failed. {error}.", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"Infrastructure or execution failure during {phase} (exit {error.returncode}).",
              file=sys.stderr)
        return 2
    except Exception as error:
        failure = ("Invalid candidate report" if isinstance(error, ValueError) and phase == "strict candidate validation"
                   else "Infrastructure or execution failure")
        print(f"{failure} during {phase} ({type(error).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
