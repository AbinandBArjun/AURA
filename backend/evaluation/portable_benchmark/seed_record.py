"""The versioned portable seed record and its runtime mappings."""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from evaluation.portable_benchmark.schema import Digest, PassageID, Slug, StrictModel, VersionOne
from evaluation.portable_benchmark.seed_infra import ResourceIdentity


RuntimeID = Annotated[int, Field(gt=0)]
PointID = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class PassageMapping(StrictModel):
    passage_id: PassageID
    chunk_id: RuntimeID
    vector_id: PointID | None
    vector_sha256: Digest | None

    @model_validator(mode="after")
    def complete_vector(self) -> Self:
        if (self.vector_id is None) != (self.vector_sha256 is None):
            raise ValueError("A vector ID and digest must be recorded together")
        return self


class DocumentMapping(StrictModel):
    document_key: Slug
    document_id: RuntimeID
    passages: Annotated[list[PassageMapping], Field(min_length=1)]

    @model_validator(mode="after")
    def matching_document(self) -> Self:
        if any(item.passage_id.split(":")[1] != self.document_key for item in self.passages):
            raise ValueError("Passage ID does not belong to its mapped document")
        return self


class SeedRecord(StrictModel):
    schema_version: VersionOne
    corpus_fingerprint: Digest
    corpus_version: str
    extraction_protocol: str
    chunking_protocol: str
    protocol_fingerprint: Digest
    passage_id_scheme: Literal["p1"]
    alembic_revision: str
    model_id: str
    model_revision: Revision
    model_manifest_sha256: Digest
    vector_dimensions: Literal[384]
    resources: ResourceIdentity
    owner_id: RuntimeID
    workspace_id: RuntimeID
    documents: Annotated[list[DocumentMapping], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_mappings(self) -> Self:
        passages = [item for document in self.documents for item in document.passages]
        values = (
            ("logical document", [item.document_key for item in self.documents]),
            ("database document", [item.document_id for item in self.documents]),
            ("logical passage", [item.passage_id for item in passages]),
            ("database chunk", [item.chunk_id for item in passages]),
            ("vector", [item.vector_id for item in passages if item.vector_id is not None]),
        )
        for label, ids in values:
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} mapping")
        if any(item.passage_id.split(":")[3] != self.protocol_fingerprint for item in passages):
            raise ValueError("Passage ID does not match the recorded protocol")
        return self
