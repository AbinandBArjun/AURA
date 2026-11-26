"""Versioned portable retrieval protocol, report schema, and safe publication."""

from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
import platform
import statistics
from math import isclose
from time import perf_counter
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, ValidationInfo, model_validator

from evaluation.metrics.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from evaluation.portable_benchmark.corpus import decode_json, json_digest, protocol_fingerprint
from evaluation.portable_benchmark.schema import Category, CorpusManifest, Digest, NonemptyText, PassageID, PortableDataset, Slug, StrictModel, VersionOne


Mode = Literal["ranking-only", "quality"]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Latency = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Revision = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]
Score = Annotated[float, Field(allow_inf_nan=False)]


class Configuration(StrictModel):
    name: str
    mode: Literal["dense", "hybrid", "hybrid_rerank"]
    candidate_pool_size: PositiveInt | None = None

    @model_validator(mode="after")
    def correct_pool_and_name(self) -> Self:
        if self.mode == "dense":
            if self.name != "dense" or self.candidate_pool_size is not None:
                raise ValueError("Dense has no candidate pool")
        elif self.candidate_pool_size is None:
            raise ValueError("Hybrid retrieval requires a candidate pool")
        elif self.name != (
            "hybrid" if self.mode == "hybrid" else f"hybrid_rerank_{self.candidate_pool_size}"
        ):
            raise ValueError("Retrieval profile name does not match its configuration")
        return self


def configurations(top_k, corpus_size, hybrid_pool=None, rerank_pools=(5, 10, 20)):
    if type(top_k) is not int or not 1 <= top_k <= corpus_size:
        raise ValueError("Top-k must be a positive integer within the retrievable corpus")
    if hybrid_pool is None:
        hybrid_pool = min(corpus_size, max(20, 4 * top_k))
    if len(rerank_pools) != 3 or any(
        type(pool) is not int or not top_k <= pool <= corpus_size
        for pool in (hybrid_pool, *rerank_pools)
    ):
        raise ValueError("Each candidate pool must be an integer between top-k and the corpus size")
    if list(rerank_pools) != sorted(set(rerank_pools)):
        raise ValueError("Supply three distinct reranking pools in increasing order")
    return [
        Configuration(name="dense", mode="dense", candidate_pool_size=None),
        Configuration(name="hybrid", mode="hybrid", candidate_pool_size=hybrid_pool),
        *[
            Configuration(name=f"hybrid_rerank_{pool}", mode="hybrid_rerank", candidate_pool_size=pool)
            for pool in rerank_pools
        ],
    ]


class Quality(StrictModel):
    precision_at_k: UnitFloat
    recall_at_k: UnitFloat
    mrr_at_k: UnitFloat
    ndcg_at_k: UnitFloat


class CategoryQuality(Quality):
    query_count: PositiveInt


class LatencySummary(StrictModel):
    mean_ms: Latency
    median_ms: Latency
    p95_ms: Latency


class RankedPassage(StrictModel):
    passage_id: PassageID
    score: Score


class QueryResult(StrictModel):
    query_id: Slug
    category: Category
    ranking: list[RankedPassage]
    latency_ms: Latency
    quality: Quality | None = None

    @model_validator(mode="after")
    def unique_ranking(self) -> Self:
        if len({item.passage_id for item in self.ranking}) != len(self.ranking):
            raise ValueError("Duplicate passage in ranking")
        return self


class ConfigurationResult(StrictModel):
    configuration: Configuration
    queries: Annotated[list[QueryResult], Field(min_length=1)]
    latency: LatencySummary
    quality: Quality | None = None
    category_quality: dict[Category, CategoryQuality] | None = None


class ModelIdentity(StrictModel):
    repo_id: Literal["sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"]
    revision: Revision
    manifest_sha256: Digest


class Runtime(StrictModel):
    python: str
    system: str
    architecture: str
    torch: str
    transformers: str
    sentence_transformers: str
    qdrant_client: str
    rank_bm25: str
    numpy: str
    tiktoken: str
    pymupdf: str


@dataclass(frozen=True)
class ReportContext:
    """Expectations prepared from independently checked benchmark inputs."""

    manifest: CorpusManifest
    passage_ids: frozenset[str]
    dataset: PortableDataset
    mode: Mode
    top_k: int
    profiles: tuple[Configuration, ...]
    embedding: ModelIdentity
    reranker: ModelIdentity

    def __post_init__(self):
        validate_mode(self.dataset, self.mode)
        if self.dataset.corpus_fingerprint != self.manifest.corpus_fingerprint:
            raise ValueError("Trusted dataset and corpus fingerprints disagree")
        if [query.query_id for query in self.dataset.queries] != list(dict.fromkeys(
            query.query_id for query in self.dataset.queries
        )):
            raise ValueError("Trusted dataset contains duplicate queries")
        for query in self.dataset.queries:
            if any(item.passage_id not in self.passage_ids for item in query.judgments):
                raise ValueError("Trusted judgments name a passage outside the corpus")
        if len(self.profiles) != 5 or self.profiles != tuple(configurations(
            self.top_k, len(self.passage_ids), self.profiles[1].candidate_pool_size,
            tuple(profile.candidate_pool_size for profile in self.profiles[2:]),
        )):
            raise ValueError("Trusted configuration is invalid")


QUALITY_TOLERANCE = 1e-12


def same_quality(reported, expected):
    return reported is not None and all(
        isclose(getattr(reported, key), getattr(expected, key), rel_tol=0, abs_tol=QUALITY_TOLERANCE)
        for key in Quality.model_fields
    )

class BenchmarkReport(StrictModel):
    schema_version: VersionOne
    evaluation_protocol: Literal["aura-portable-retrieval-v1"]
    passage_id_scheme: Literal["p1"]
    passage_protocol_sha256: Digest
    corpus_name: NonemptyText
    corpus_version: NonemptyText
    corpus_fingerprint: Digest
    dataset_fingerprint: Digest
    dataset_review_status: Literal["draft", "reviewed"]
    mode: Mode
    top_k: PositiveInt
    retrievable_passage_count: PositiveInt
    embedding: ModelIdentity
    reranker: ModelIdentity
    runtime: Runtime
    limitations: Annotated[list[str], Field(min_length=1)]
    results: Annotated[list[ConfigurationResult], Field(min_length=5, max_length=5)]

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs) -> Self:
        return cls.model_validate(decode_json(json_data), **kwargs)

    @model_validator(mode="after")
    def consistent_results(self, info: ValidationInfo) -> Self:
        if info.mode == "json":
            # Generic Pydantic JSON adapters bypass the strict byte parser.
            raise ValueError("Use BenchmarkReport.model_validate_json or validate_report for report JSON")
        context = info.context
        if not isinstance(context, ReportContext):
            raise ValueError("Trusted portable report validation context is required")
        manifest = context.manifest
        dataset = context.dataset
        if (
            self.corpus_name != manifest.corpus_name or self.corpus_version != manifest.corpus_version
            or self.corpus_fingerprint != manifest.corpus_fingerprint
            or self.passage_id_scheme != manifest.passage_id_scheme
            or self.passage_protocol_sha256 != protocol_fingerprint(manifest)
            or self.dataset_fingerprint != json_digest(dataset.model_dump())
            or self.dataset_review_status != dataset.review_status
            or self.retrievable_passage_count != len(context.passage_ids)
        ):
            raise ValueError("Report corpus, passage protocol, or dataset does not match trusted inputs")
        if self.embedding != context.embedding or self.reranker != context.reranker:
            raise ValueError("Report model identities do not match trusted inputs")
        if (
            self.mode != context.mode or self.top_k != context.top_k
            or tuple(result.configuration for result in self.results) != context.profiles
        ):
            raise ValueError("Report retrieval configurations do not match trusted inputs")

        expected_queries = [(query.query_id, query.category) for query in dataset.queries]
        quality = context.mode == "quality"
        for result in self.results:
            if [(query.query_id, query.category) for query in result.queries] != expected_queries:
                raise ValueError("Report query IDs and categories do not match the trusted dataset")
            if not quality and {"quality", "category_quality"} & result.model_fields_set:
                raise ValueError("Ranking-only reports cannot contain quality fields")
            verified = []
            by_category = defaultdict(list)
            for query, expected_query in zip(result.queries, dataset.queries, strict=True):
                ids = [item.passage_id for item in query.ranking]
                if len(ids) != context.top_k or not set(ids) <= context.passage_ids:
                    raise ValueError("Incorrect ranking length or unknown logical passage")
                if query.ranking != sorted(query.ranking, key=lambda item: (-item.score, item.passage_id)):
                    raise ValueError("Report ranking has incorrect score or logical tie order")
                if not quality and "quality" in query.model_fields_set:
                    raise ValueError("Ranking-only reports cannot contain query quality fields")
                if quality:
                    expected_quality = quality_scores(ids, expected_query.judgments, context.top_k)
                    if not same_quality(query.quality, expected_quality):
                        raise ValueError("Reported query metrics disagree with the reviewed judgments")
                    verified.append(expected_quality)
                    by_category[expected_query.category].append(expected_quality)
            if quality:
                if not same_quality(result.quality, Quality(**mean_quality(verified))):
                    raise ValueError("Reported overall metrics disagree with the recomputed query metrics")
                if result.category_quality is None or result.category_quality.keys() != by_category.keys():
                    raise ValueError("Report categories do not match the trusted dataset")
                for category, rows in by_category.items():
                    reported = result.category_quality[category]
                    if reported.query_count != len(rows) or not same_quality(reported, Quality(**mean_quality(rows))):
                        raise ValueError("Reported category denominator or metrics disagree with the trusted dataset")
        return self


def validate_report(data, context: ReportContext):
    """Validate stored report bytes against independently established expectations."""
    return BenchmarkReport.model_validate_json(data, context=context)

def quality_scores(ranking, judgments, k):
    grades = {item.passage_id: item.grade for item in judgments}
    if len(grades) != len(judgments) or any(
        item.review_status != "reviewed" or not item.reviewer_id or item.assessor_id == item.reviewer_id
        for item in judgments
    ) or not any(grade > 0 for grade in grades.values()):
        raise ValueError("Quality evaluation requires unique reviewed judgments with a positive grade for each query")
    return Quality(
        precision_at_k=precision_at_k(ranking, grades, k),
        recall_at_k=recall_at_k(ranking, grades, k),
        mrr_at_k=reciprocal_rank(ranking[:k], grades),
        ndcg_at_k=ndcg_at_k(ranking, grades, k),
    )


def validate_mode(dataset, mode):
    if mode not in ("ranking-only", "quality") or (
        (mode == "quality") != (dataset.review_status == "reviewed")
    ):
        raise ValueError("Use ranking-only for draft datasets and quality for reviewed datasets")
    if mode == "quality":
        for query in dataset.queries:
            quality_scores([], query.judgments, 1)


def mean_quality(rows):
    return {
        key: statistics.mean(getattr(item, key) for item in rows)
        for key in Quality.model_fields
    }


def evaluate(service, session, workspace_id, mapping, dataset, mode, top_k, profiles):
    validate_mode(dataset, mode)
    eligible = {payload["passage_id"] for payload in mapping.points.values()}
    output = []
    for configuration in profiles:
        queries = []
        by_category = defaultdict(list)
        for item in dataset.queries:
            started = perf_counter()
            returned = service.search(
                db=session, workspace_id=workspace_id, query=item.query, limit=top_k,
                mode=configuration.mode, candidate_pool_size=configuration.candidate_pool_size,
            )
            latency = (perf_counter() - started) * 1000
            ranking = [RankedPassage(passage_id=row["chunk_id"], score=row["score"]) for row in returned]
            ids = [row.passage_id for row in ranking]
            if len(ids) != top_k or len(set(ids)) != len(ids) or not set(ids) <= eligible:
                raise ValueError("Retrieval returned an incomplete, unknown, or duplicate logical ranking")
            result = QueryResult(
                query_id=item.query_id, category=item.category, ranking=ranking, latency_ms=latency,
                quality=quality_scores(ids, item.judgments, top_k) if mode == "quality" else None,
            )
            queries.append(result)
            by_category[item.category].append(result)
        latencies = [row.latency_ms for row in queries]
        output.append(ConfigurationResult(
            configuration=configuration, queries=queries,
            latency=LatencySummary(
                mean_ms=statistics.mean(latencies), median_ms=statistics.median(latencies),
                p95_ms=statistics.quantiles(latencies, n=100, method="inclusive")[94] if len(latencies) > 1 else latencies[0],
            ),
            quality=Quality(**mean_quality([query.quality for query in queries])) if mode == "quality" else None,
            category_quality={
                key: CategoryQuality(query_count=len(rows), **mean_quality([query.quality for query in rows]))
                for key, rows in sorted(by_category.items())
            } if mode == "quality" else None,
        ))
    return output


def model_identity(manifest):
    return ModelIdentity(repo_id=manifest["repo_id"], revision=manifest["revision"], manifest_sha256=json_digest(manifest))


def runtime():
    return Runtime(
        python=platform.python_version(), system=platform.system(), architecture=platform.machine(),
        torch=version("torch"), transformers=version("transformers"),
        sentence_transformers=version("sentence-transformers"), qdrant_client=version("qdrant-client"),
        rank_bm25=version("rank-bm25"), numpy=version("numpy"), tiktoken=version("tiktoken"),
        pymupdf=version("PyMuPDF"),
    )
