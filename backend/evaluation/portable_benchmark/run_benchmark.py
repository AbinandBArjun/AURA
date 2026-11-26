"""Run the portable retrieval benchmark against an owned, already-seeded stack."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import file_digest
import logging
import os
from pathlib import Path
import re
from runpy import run_path
import sys
from warnings import catch_warnings, simplefilter
from tempfile import TemporaryDirectory

from evaluation.portable_benchmark.corpus import (
    CORPUS_DIR, json_digest, load_corpus, load_dataset, passage_catalog, protocol_fingerprint, read_json,
)
from evaluation.portable_benchmark.ranking import run_against_seed
from evaluation.portable_benchmark.ranking_report import (
    BenchmarkReport, ReportContext, configurations, evaluate, model_identity, runtime, validate_mode,
)
from evaluation.portable_benchmark.report_output import output_destination, publish
from evaluation.portable_benchmark.reranker_assets import (
    load_manifest as load_reranker_manifest, local_reranker, verified_reranker,
)
from evaluation.portable_benchmark.seed_assets import (
    TEST_MODEL_PREPARATION, LocalEmbeddingService, prepare_corpus, verified_model, verify_passage_ids,
)
from evaluation.portable_benchmark.seed_infra import resolve_targets


DRAFT_DATASET_SHA256 = "cc6d83fe50eb3a785b74295651f0c7ee48d384bc70a573a02a64478128bbd756"
LIMITATIONS = [
    "Exact dense scoring visits all verified passages before logical-ID tie-breaking.",
    "Candidate cutoffs and final score ties use logical IDs. Floating-point scores may vary by runtime.",
    "Latency includes database and Docker exec transport overhead and has no quality-gate role.",
    "The report uses a read-only SQL snapshot and checks the Qdrant state before and after retrieval.",
    "This report is separate from the historical benchmark and is not an approved quality baseline.",
]


class BenchmarkFailure(RuntimeError):
    """A safe message assembled by the runner, without exception details."""


@contextmanager
def phase(message, *, output=False):
    try:
        yield
    except FileExistsError:
        if output:
            raise BenchmarkFailure("Output already exists. Choose a new report filename.") from None
        raise BenchmarkFailure(message) from None
    except Exception:
        raise BenchmarkFailure(message) from None


def dataset_fingerprint(dataset):
    return json_digest(dataset.model_dump())


def select_dataset(manifest, catalog, mode, path=None, expected_sha256=None):
    draft = load_dataset(CORPUS_DIR, manifest, catalog)
    if dataset_fingerprint(draft) != DRAFT_DATASET_SHA256:
        raise ValueError("Checked-in draft dataset fingerprint mismatch")
    if (path is None) != (expected_sha256 is None):
        raise ValueError("Supply --dataset and --dataset-sha256 together")
    if path is None:
        selected = draft
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Dataset SHA-256 must be 64 lowercase hexadecimal digits")
        selected = load_dataset(CORPUS_DIR, manifest, catalog, dataset_path=path)
        if dataset_fingerprint(selected) != expected_sha256:
            raise ValueError("Dataset fingerprint mismatch")
        draft_queries = [(q.query_id, q.category, q.query) for q in draft.queries]
        if [(q.query_id, q.category, q.query) for q in selected.queries] != draft_queries:
            raise ValueError("Alternate dataset must cover the checked-in query IDs, text, and categories in the same order")
    validate_mode(selected, mode)
    return selected


def load_report_context(mode, top_k=5, hybrid_pool=None, rerank_pools=(5, 10, 20),
                        dataset_path=None, expected_dataset_sha256=None):
    """Build report expectations without using any fields from a report under review."""
    with phase("Corpus validation failed. Check the committed sources, manifest and passage catalog."):
        manifest, sources = load_corpus(CORPUS_DIR)
        catalog = passage_catalog(manifest, sources)
        verify_passage_ids(
            [passage.passage_id for passage in catalog],
            read_json(CORPUS_DIR / "passage_ids.json"), manifest.corpus_fingerprint,
        )
        passage_ids = frozenset(passage.passage_id for passage in catalog if passage.chunk.retrievable)
    with phase("Dataset validation failed. Use ranking-only for draft datasets and quality for reviewed datasets. "
               "Check the canonical digest, complete query catalog and judgments."):
        dataset = select_dataset(manifest, catalog, mode, dataset_path, expected_dataset_sha256)
    with phase("Invalid retrieval configuration. Check top-k and the candidate pool sizes."):
        profiles = tuple(configurations(top_k, len(passage_ids), hybrid_pool, rerank_pools))
    with phase("Pinned model manifests failed verification. Check the committed benchmark model manifests."):
        embedding = model_identity(run_path(str(TEST_MODEL_PREPARATION))["load_manifest"]())
        reranker = model_identity(load_reranker_manifest())
    return ReportContext(manifest, passage_ids, dataset, mode, top_k, profiles, embedding, reranker)


def asset_problem(directory, manifest):
    """Return only names already allowlisted in the version-controlled manifest."""
    if not directory.is_dir() or directory.is_symlink() or directory.is_junction():
        return "model directory is missing or unsafe"
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or path.is_junction() or not path.is_file():
            return f"missing or unsafe asset {name}"
        with path.open("rb") as file:
            actual = file_digest(file, "sha256").hexdigest()
        if actual != expected:
            return f"checksum mismatch for {name}"
    return "model directory contains unexpected assets or could not be verified"


def checked_assets(verify, directory, load_manifest, identity):
    try:
        path, manifest = verify(directory)
        if model_identity(manifest) == identity:
            return path, manifest
        issue = "model identity disagrees with the pinned manifest"
    except Exception:
        try:
            issue = asset_problem(directory, load_manifest())
        except Exception:
            issue = "model assets could not be read safely"
    raise BenchmarkFailure(f"{identity.repo_id}: {issue}. Prepare the pinned assets again.") from None


def run(args):
    with phase("Unsafe report destination. Choose an existing local directory and a new .json filename.", output=True):
        output = output_destination(args.output)
    context = load_report_context(
        args.mode, args.top_k, args.hybrid_pool, args.rerank_pools, args.dataset, args.dataset_sha256,
    )
    manifest, dataset = context.manifest, context.dataset
    with phase("Corpus preparation failed. Check the portable benchmark installation and extraction assets."):
        with TemporaryDirectory(prefix="aura-portable-ranking-") as temporary:
            corpus = prepare_corpus(CORPUS_DIR, Path(temporary) / "pdfs")
    with phase("Benchmark infrastructure rejected. Check the local Docker context, dedicated project, ports and ownership."):
        targets = resolve_targets(args.docker_context, args.project, args.postgres_port, args.qdrant_port)
    embedding_path, embedding_manifest = checked_assets(
        verified_model, args.embedding_model_dir,
        lambda: run_path(str(TEST_MODEL_PREPARATION))["load_manifest"](), context.embedding,
    )
    reranker_path, _ = checked_assets(
        verified_reranker, args.reranker_model_dir, load_reranker_manifest, context.reranker,
    )
    with phase("Retrieval or readiness verification failed. Check the seeded corpus, service identity and offline model assets."):
        results = run_against_seed(
            targets, corpus, embedding_manifest,
            lambda: LocalEmbeddingService(embedding_path),
            lambda: local_reranker(reranker_path),
            lambda service, session, workspace_id, mapping: evaluate(
                service, session, workspace_id, mapping, dataset, context.mode, context.top_k, context.profiles,
            ),
        )
    with phase("Report integrity validation failed. Check the corpus, dataset, retrieval results and runtime metadata."):
        # Construct the report through the same trusted validation used by readers.
        report = BenchmarkReport.model_validate(dict(
            schema_version=1, evaluation_protocol="aura-portable-retrieval-v1",
            passage_id_scheme=manifest.passage_id_scheme, passage_protocol_sha256=protocol_fingerprint(manifest),
            corpus_name=manifest.corpus_name, corpus_version=manifest.corpus_version,
            corpus_fingerprint=manifest.corpus_fingerprint, dataset_fingerprint=dataset_fingerprint(dataset),
            dataset_review_status=dataset.review_status, mode=context.mode, top_k=context.top_k,
            retrievable_passage_count=len(context.passage_ids), embedding=context.embedding,
            reranker=context.reranker, runtime=runtime(), limitations=LIMITATIONS,
            results=[result.model_dump(exclude_none=True) for result in results],
        ), context=context)
    with phase("Report publication failed. Check directory ownership, filesystem support and available space.", output=True):
        publish(report, output, context)
    return report


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "Invalid benchmark arguments. Use --help for accepted options.\n")


def parse_args(argv=None):
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--postgres-port", type=int, required=True)
    parser.add_argument("--qdrant-port", type=int, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("ranking-only", "quality"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hybrid-pool", type=int)
    parser.add_argument("--rerank-pools", type=int, nargs=3, default=(5, 10, 20))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Library warnings and progress can include absolute paths or service details.
    previous = logging.root.manager.disable
    logging.disable(sys.maxsize)
    try:
        with (open(os.devnull, "w", encoding="utf-8") as quiet, redirect_stderr(quiet),
              redirect_stdout(quiet), catch_warnings()):
            simplefilter("ignore")
            report = run(args)
    except BenchmarkFailure as error:
        print(f"Portable benchmark failed. {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Portable benchmark failed. Check the benchmark installation and inputs.", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous)
    print(f"Published a {report.mode} report with {len(report.results)} retrieval configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
