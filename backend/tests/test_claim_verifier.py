import pytest

from app.services.claim_verifier import (
    ClaimCitationError,
    validate_claim_citations,
)


@pytest.fixture
def evidence_map():
    return {
        "S1": {
            "document_id": 1,
            "filename": "paper1.pdf",
            "chunk_id": 10,
            "page_start": 1,
            "page_end": 2,
            "score": 0.85,
        },
        "S2": {
            "document_id": 2,
            "filename": "paper2.pdf",
            "chunk_id": 20,
            "page_start": 3,
            "page_end": 4,
            "score": 0.75,
        },
    }


def test_valid_claim_citations(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "RAG combines retrieval and generation.",
        "citation_ids": ["S1"],
    }]

    result = validate_claim_citations(claims, evidence_map)

    assert len(result) == 1
    assert result[0]["claim_id"] == "C1"
    assert result[0]["citations"][0]["citation_id"] == "S1"
    assert result[0]["citations"][0]["source"]["document_id"] == 1


def test_fabricated_citation_is_rejected(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "An unsupported claim.",
        "citation_ids": ["S99"],
    }]

    with pytest.raises(ClaimCitationError, match="Unknown citation ID"):
        validate_claim_citations(claims, evidence_map)


def test_invalid_citation_format_is_rejected(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "A claim.",
        "citation_ids": ["SOURCE1"],
    }]

    with pytest.raises(ClaimCitationError, match="Invalid citation ID"):
        validate_claim_citations(claims, evidence_map)


def test_duplicate_citations_are_removed(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "A claim.",
        "citation_ids": ["S1", "S1", "S2"],
    }]

    result = validate_claim_citations(claims, evidence_map)

    citation_ids = [
        citation["citation_id"]
        for citation in result[0]["citations"]
    ]

    assert citation_ids == ["S1", "S2"]


def test_model_cannot_override_source_metadata(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "A claim.",
        "citation_ids": ["S1"],
        "document_id": 999,
        "filename": "fabricated.pdf",
        "page_start": 999,
    }]

    result = validate_claim_citations(claims, evidence_map)

    source = result[0]["citations"][0]["source"]

    assert source["document_id"] == 1
    assert source["filename"] == "paper1.pdf"
    assert source["page_start"] == 1


def test_claim_without_citations_is_rejected(evidence_map):
    claims = [{
        "claim_id": "C1",
        "text": "A claim without evidence.",
        "citation_ids": [],
    }]

    with pytest.raises(ClaimCitationError, match="must contain citations"):
        validate_claim_citations(claims, evidence_map)


def test_duplicate_claim_ids_are_rejected(evidence_map):
    claims = [
        {
            "claim_id": "C1",
            "text": "First claim.",
            "citation_ids": ["S1"],
        },
        {
            "claim_id": "C1",
            "text": "Second claim.",
            "citation_ids": ["S2"],
        },
    ]

    with pytest.raises(ClaimCitationError, match="Duplicate claim ID"):
        validate_claim_citations(claims, evidence_map)


def test_incomplete_source_metadata_is_rejected(evidence_map):
    evidence_map["S3"] = {
        "document_id": 3,
        "filename": "incomplete.pdf",
    }

    claims = [{
        "claim_id": "C1",
        "text": "A claim.",
        "citation_ids": ["S3"],
    }]

    with pytest.raises(
        ClaimCitationError,
        match="Incomplete evidence metadata",
    ):
        validate_claim_citations(claims, evidence_map)
