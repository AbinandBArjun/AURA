from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


VersionOne = Annotated[int, Field(ge=1, le=1)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SourcePath = Annotated[str, StringConstraints(pattern=r"^sources/[a-z0-9]+(-[a-z0-9]+)*\.txt$")]
PassageID = Annotated[
    str,
    StringConstraints(pattern=r"^p1:[a-z0-9]+(-[a-z0-9]+)*:(0|[1-9][0-9]*):[0-9a-f]{64}:[0-9a-f]{64}$"),
]
Domain = Literal["artificial-intelligence", "cybersecurity", "economics", "software-engineering"]
Category = Literal[
    "artificial-intelligence", "cybersecurity", "economics", "software-engineering", "cross-domain"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DocumentSpec(StrictModel):
    key: Slug
    title: NonemptyText
    domain: Domain
    source_path: SourcePath
    source_sha256: Digest
    provenance: Literal["original-synthetic-for-aura"]
    redistribution: Literal["no-grant-recorded"]


class ExtractionSpec(StrictModel):
    protocol: Literal["text-pdf-v1"]
    source_encoding: Literal["utf-8-ascii-lf-v1"]
    line_width: Literal[88]
    lines_per_page: Literal[50]


class ChunkingSpec(StrictModel):
    protocol: Literal["aura-section-page-chunks-v1"]
    tokenizer: Literal["cl100k_base"]
    tokenizer_package_version: Literal["0.14.0"]
    chunk_size: Literal[500]
    chunk_overlap: Literal[75]


class CorpusManifest(StrictModel):
    schema_version: VersionOne
    corpus_name: NonemptyText
    corpus_version: Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
    extraction: ExtractionSpec
    chunking: ChunkingSpec
    passage_id_scheme: Literal["p1"]
    corpus_fingerprint: Digest
    documents: Annotated[list[DocumentSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_documents(self) -> Self:
        keys = [document.key for document in self.documents]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate document key")
        paths = [document.source_path for document in self.documents]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate document source path")
        return self


class Judgment(StrictModel):
    passage_id: PassageID
    grade: Annotated[int, Field(ge=0, le=3)]
    method: Literal["human"]
    assessor_id: Slug
    review_status: Literal["pending", "reviewed"]
    reviewer_id: Slug | None = None

    @model_validator(mode="after")
    def independent_review(self) -> Self:
        if self.review_status == "reviewed":
            if self.reviewer_id is None or self.reviewer_id == self.assessor_id:
                raise ValueError("Reviewed judgment requires a different reviewer")
        elif self.reviewer_id is not None:
            raise ValueError("Pending judgment cannot name a reviewer")
        return self


class Query(StrictModel):
    query_id: Slug
    category: Category
    query: NonemptyText
    judgments: list[Judgment]

    @model_validator(mode="after")
    def unique_judgments(self) -> Self:
        ids = [judgment.passage_id for judgment in self.judgments]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate passage judgment in {self.query_id}")
        return self


class PortableDataset(StrictModel):
    schema_version: VersionOne
    corpus_fingerprint: Digest
    review_status: Literal["draft", "reviewed"]
    queries: Annotated[list[Query], Field(min_length=1)]

    @model_validator(mode="after")
    def query_integrity(self) -> Self:
        ids = [query.query_id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate query ID")
        if self.review_status == "reviewed":
            for query in self.queries:
                if not any(judgment.grade > 0 for judgment in query.judgments):
                    raise ValueError(f"Reviewed query requires a positive grade: {query.query_id}")
                if any(judgment.review_status != "reviewed" for judgment in query.judgments):
                    raise ValueError(f"Reviewed dataset contains pending judgments: {query.query_id}")
        return self
