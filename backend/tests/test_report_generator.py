import pytest

from app.agents.report_generator import (
    ReportGeneratorError,
    ReportGeneratorNode,
)


@pytest.fixture
def valid_state():
    return {
        "question": "What are the benefits of hybrid retrieval?",
        "findings": "Hybrid retrieval combines dense and sparse search.",
        "evidence": [
            {
                "citation_id": "S1",
                "document_id": 1,
                "filename": "research.pdf",
                "chunk_id": 10,
                "text": "Dense and sparse retrieval provide complementary signals.",
                "page_start": 2,
                "page_end": 3,
                "score": 0.92,
                "research_query": "hybrid retrieval",
            },
            {
                "citation_id": "S2",
                "document_id": 2,
                "filename": "evaluation.pdf",
                "chunk_id": 20,
                "text": "Hybrid retrieval improves retrieval coverage.",
                "page_start": 5,
                "page_end": 5,
                "score": 0.88,
                "research_query": "retrieval evaluation",
            },
        ],
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "Hybrid retrieval combines dense and sparse search.",
                "evidence_ids": ["S1"],
            },
            {
                "claim_id": "C2",
                "claim_text": "Hybrid retrieval improves retrieval coverage.",
                "evidence_ids": ["S2"],
            },
            {
                "claim_id": "C3",
                "claim_text": "The approach is proven optimal in every domain.",
                "evidence_ids": ["S1", "S2"],
            },
            {
                "claim_id": "C4",
                "claim_text": "The evidence does not establish a universal result.",
                "evidence_ids": ["S1"],
            },
        ],
        "verification_results": [
            {
                "claim_id": "C1",
                "status": "supported",
                "rationale": "The evidence directly describes both retrieval methods.",
                "evidence_ids": ["S1"],
            },
            {
                "claim_id": "C2",
                "status": "partially_supported",
                "rationale": "The source reports improved coverage but not every condition.",
                "evidence_ids": ["S2"],
            },
            {
                "claim_id": "C3",
                "status": "contradicted",
                "rationale": "The evidence does not establish universal optimality.",
                "evidence_ids": ["S1", "S2"],
            },
            {
                "claim_id": "C4",
                "status": "insufficient_evidence",
                "rationale": "The supplied evidence cannot establish this conclusion.",
                "evidence_ids": [],
            },
        ],
    }


def test_generates_report_with_all_verification_statuses(valid_state):
    node = ReportGeneratorNode()

    result = node(valid_state)

    report = result["report"]

    assert "# Research Report" in report
    assert "## Research Question" in report
    assert "## Executive Summary" in report
    assert "## Claim Verification" in report
    assert "## Limitations and Caveats" in report
    assert "## Sources" in report

    assert "**Verification status:** Supported" in report
    assert "**Verification status:** Partially supported" in report
    assert "**Verification status:** Contradicted" in report
    assert "**Verification status:** Insufficient evidence" in report


def test_preserves_claim_and_verification_evidence(valid_state):
    report = ReportGeneratorNode()(valid_state)["report"]

    assert "**Claim evidence:** [S1]" in report
    assert "**Verification evidence:** [S1]" in report
    assert "**Verification evidence:** No evidence identified." in report


def test_preserves_source_metadata(valid_state):
    report = ReportGeneratorNode()(valid_state)["report"]

    assert "[S1]** research.pdf" in report
    assert "Document 1" in report
    assert "Chunk 10" in report
    assert "Pages 2-3" in report

    assert "[S2]** evaluation.pdf" in report
    assert "Page 5" in report


def test_includes_only_sources_referenced_by_claims(valid_state):
    valid_state["evidence"].append({
        "citation_id": "S3",
        "document_id": 3,
        "filename": "unused.pdf",
        "chunk_id": 30,
        "text": "This evidence is not referenced.",
        "page_start": 1,
        "page_end": 1,
    })

    report = ReportGeneratorNode()(valid_state)["report"]

    assert "unused.pdf" not in report


def test_rejects_missing_verification_result(valid_state):
    valid_state["verification_results"].pop()

    with pytest.raises(ReportGeneratorError, match="Missing verification"):
        ReportGeneratorNode()(valid_state)


def test_rejects_duplicate_claim_ids(valid_state):
    valid_state["claims"].append(valid_state["claims"][0].copy())

    with pytest.raises(ReportGeneratorError, match="Duplicate claim ID"):
        ReportGeneratorNode()(valid_state)


def test_rejects_duplicate_verification_results(valid_state):
    valid_state["verification_results"].append(
        valid_state["verification_results"][0].copy()
    )

    with pytest.raises(ReportGeneratorError, match="Duplicate verification"):
        ReportGeneratorNode()(valid_state)


def test_rejects_unknown_evidence_ids(valid_state):
    valid_state["claims"][0]["evidence_ids"] = ["S99"]

    with pytest.raises(ReportGeneratorError, match="unknown evidence"):
        ReportGeneratorNode()(valid_state)


def test_rejects_verification_evidence_not_assigned_to_claim(valid_state):
    valid_state["verification_results"][0]["evidence_ids"] = ["S2"]

    with pytest.raises(
        ReportGeneratorError,
        match="not assigned to claim",
    ):
        ReportGeneratorNode()(valid_state)


def test_rejects_invalid_verification_status(valid_state):
    valid_state["verification_results"][0]["status"] = "unverified"

    with pytest.raises(ReportGeneratorError, match="Invalid verification status"):
        ReportGeneratorNode()(valid_state)


def test_rejects_non_string_verification_status(valid_state):
    valid_state["verification_results"][0]["status"] = []

    with pytest.raises(ReportGeneratorError, match="Invalid verification status"):
        ReportGeneratorNode()(valid_state)


def test_rejects_missing_evidence_for_non_insufficient_status(valid_state):
    valid_state["verification_results"][0]["evidence_ids"] = []

    with pytest.raises(ReportGeneratorError, match="evidence is required"):
        ReportGeneratorNode()(valid_state)


def test_rejects_invalid_source_metadata(valid_state):
    valid_state["evidence"][0]["document_id"] = 0

    with pytest.raises(ReportGeneratorError, match="source identifiers"):
        ReportGeneratorNode()(valid_state)


def test_rejects_invalid_page_range(valid_state):
    valid_state["evidence"][0]["page_start"] = 5
    valid_state["evidence"][0]["page_end"] = 2

    with pytest.raises(ReportGeneratorError, match="page range"):
        ReportGeneratorNode()(valid_state)


def test_rejects_empty_findings(valid_state):
    valid_state["findings"] = " "

    with pytest.raises(ReportGeneratorError, match="Research findings"):
        ReportGeneratorNode()(valid_state)


def test_rejects_malformed_workflow_state(valid_state):
    valid_state["claims"] = "not a list"

    with pytest.raises(ReportGeneratorError, match="Claims must be a list"):
        ReportGeneratorNode()(valid_state)


def test_handles_empty_claims_and_evidence():
    state = {
        "question": "What did the research establish?",
        "findings": "No usable evidence was retrieved.",
        "claims": [],
        "verification_results": [],
        "evidence": [],
    }

    report = ReportGeneratorNode()(state)["report"]

    assert "No verifiable claims were produced" in report
    assert "No claims could be verified" in report
    assert "No sources were linked" in report


def test_escapes_untrusted_markdown_content(valid_state):
    valid_state["question"] = (
        "Research question\n## Injected heading\n[malicious link](https://example.com)"
    )
    valid_state["findings"] = (
        "Research findings\n# Injected summary heading"
    )
    valid_state["claims"][0]["claim_text"] = (
        "Claim text\n## Injected claim heading"
    )
    valid_state["verification_results"][0]["rationale"] = (
        "Verification rationale\n## Injected rationale heading"
    )

    report = ReportGeneratorNode()(valid_state)["report"]

    assert "> Research question" in report
    assert "> \\#\\# Injected heading" in report
    assert "\\[malicious link\\]" in report
    assert "> \\# Injected summary heading" in report
    assert "> \\#\\# Injected claim heading" in report
    assert "> \\#\\# Injected rationale heading" in report
