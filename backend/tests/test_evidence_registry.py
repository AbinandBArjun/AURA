import pytest

from app.agents.evidence_registry import (
    EvidenceRegistryError,
    EvidenceRegistryNode,
)


def make_evidence(chunk_id=10):
    return {
        "document_id": 1,
        "filename": "research.pdf",
        "chunk_id": chunk_id,
        "text": "Retrieved research evidence.",
        "page_start": 1,
        "page_end": 2,
        "score": 0.85,
    }


def test_registry_assigns_sequential_citation_ids():
    registry = EvidenceRegistryNode()

    result = registry({
        "evidence": [
            make_evidence(10),
            make_evidence(11),
        ]
    })

    assert [
        item["citation_id"]
        for item in result["evidence"]
    ] == ["S1", "S2"]


def test_registry_preserves_source_metadata():
    registry = EvidenceRegistryNode()

    result = registry({
        "evidence": [make_evidence()]
    })

    evidence = result["evidence"][0]

    assert evidence["document_id"] == 1
    assert evidence["chunk_id"] == 10
    assert evidence["filename"] == "research.pdf"
    assert evidence["page_start"] == 1
    assert evidence["page_end"] == 2
    assert evidence["score"] == 0.85


def test_registry_handles_empty_evidence():
    registry = EvidenceRegistryNode()

    result = registry({"evidence": []})

    assert result["evidence"] == []


def test_registry_rejects_non_list_evidence():
    registry = EvidenceRegistryNode()

    with pytest.raises(EvidenceRegistryError, match="list"):
        registry({"evidence": "invalid"})


def test_registry_rejects_invalid_evidence_item():
    registry = EvidenceRegistryNode()

    with pytest.raises(EvidenceRegistryError, match="dictionary"):
        registry({"evidence": [None]})


def test_registry_rejects_empty_evidence_text():
    registry = EvidenceRegistryNode()

    evidence = make_evidence()
    evidence["text"] = " "

    with pytest.raises(EvidenceRegistryError, match="text"):
        registry({"evidence": [evidence]})
