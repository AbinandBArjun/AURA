import pytest

from app.agents.fact_checker import FactCheckerError, FactCheckerNode
from app.services.semantic_verifier import (
    ClaimVerificationResult,
    SemanticVerifierError,
    VerificationStatus,
)


class FakeSemanticVerifier:
    def __init__(self, status=VerificationStatus.SUPPORTED, error=None):
        self.status = status
        self.error = error
        self.calls = []

    def verify_claim(self, claim_id, claim_text, evidence):
        self.calls.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "evidence": evidence,
        })

        if self.error:
            raise self.error

        return ClaimVerificationResult(
            claim_id=claim_id,
            status=self.status,
            rationale="Assessment based on supplied evidence.",
            evidence_ids=[
                item["citation_id"] for item in evidence
            ],
        )


@pytest.fixture
def evidence():
    return [
        {
            "citation_id": "S1",
            "document_id": 1,
            "chunk_id": 10,
            "text": "Hybrid retrieval combines dense and sparse search.",
        },
        {
            "citation_id": "S2",
            "document_id": 1,
            "chunk_id": 11,
            "text": "Dense retrieval uses semantic embeddings.",
        },
    ]


@pytest.fixture
def claims():
    return [
        {
            "claim_id": "C1",
            "claim_text": "Hybrid retrieval combines dense and sparse search.",
            "evidence_ids": ["S1"],
        },
        {
            "claim_id": "C2",
            "claim_text": "Dense retrieval uses semantic embeddings.",
            "evidence_ids": ["S2"],
        },
    ]


def test_fact_checker_verifies_claims(evidence, claims):
    verifier = FakeSemanticVerifier()
    checker = FactCheckerNode(verifier)

    result = checker({
        "claims": claims,
        "evidence": evidence,
    })

    assert len(result["verification_results"]) == 2
    assert result["verification_results"][0]["claim_id"] == "C1"
    assert result["verification_results"][1]["claim_id"] == "C2"
    assert len(verifier.calls) == 2


def test_fact_checker_passes_only_referenced_evidence(evidence):
    verifier = FakeSemanticVerifier()
    checker = FactCheckerNode(verifier)

    checker({
        "claims": [{
            "claim_id": "C1",
            "claim_text": "Hybrid retrieval combines dense and sparse search.",
            "evidence_ids": ["S1"],
        }],
        "evidence": evidence,
    })

    assert verifier.calls[0]["evidence"] == [{
        "citation_id": "S1",
        "text": evidence[0]["text"],
    }]


@pytest.mark.parametrize(
    "status",
    [
        VerificationStatus.SUPPORTED,
        VerificationStatus.PARTIALLY_SUPPORTED,
        VerificationStatus.CONTRADICTED,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    ],
)
def test_fact_checker_preserves_verification_status(
    evidence,
    status,
):
    verifier = FakeSemanticVerifier(status=status)
    checker = FactCheckerNode(verifier)

    result = checker({
        "claims": [{
            "claim_id": "C1",
            "claim_text": "A research claim.",
            "evidence_ids": ["S1"],
        }],
        "evidence": evidence,
    })

    assert result["verification_results"][0]["status"] == status.value


def test_fact_checker_handles_empty_claims(evidence):
    verifier = FakeSemanticVerifier()
    checker = FactCheckerNode(verifier)

    result = checker({
        "claims": [],
        "evidence": evidence,
    })

    assert result["verification_results"] == []
    assert verifier.calls == []


def test_fact_checker_handles_empty_evidence_and_claims():
    verifier = FakeSemanticVerifier()
    checker = FactCheckerNode(verifier)

    result = checker({
        "claims": [],
        "evidence": [],
    })

    assert result["verification_results"] == []


def test_fact_checker_rejects_non_list_claims(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Claims must be a list"):
        checker({
            "claims": "invalid",
            "evidence": evidence,
        })


def test_fact_checker_rejects_non_list_evidence(claims):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Evidence must be a list"):
        checker({
            "claims": claims,
            "evidence": "invalid",
        })


@pytest.mark.parametrize("invalid_item", [None, "invalid", 123])
def test_fact_checker_rejects_invalid_evidence_items(invalid_item):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="dictionary"):
        checker({
            "claims": [],
            "evidence": [invalid_item],
        })


def test_fact_checker_rejects_duplicate_evidence_ids():
    checker = FactCheckerNode(FakeSemanticVerifier())

    item = {
        "citation_id": "S1",
        "text": "Some evidence.",
    }

    with pytest.raises(FactCheckerError, match="Duplicate evidence"):
        checker({
            "claims": [],
            "evidence": [item, item],
        })


def test_fact_checker_rejects_empty_evidence_text():
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Evidence text"):
        checker({
            "claims": [],
            "evidence": [{
                "citation_id": "S1",
                "text": " ",
            }],
        })


@pytest.mark.parametrize("invalid_claim", [None, "invalid", 123])
def test_fact_checker_rejects_invalid_claim_items(
    evidence,
    invalid_claim,
):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="dictionary"):
        checker({
            "claims": [invalid_claim],
            "evidence": evidence,
        })


def test_fact_checker_rejects_empty_claim_id(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Claim ID"):
        checker({
            "claims": [{
                "claim_id": " ",
                "claim_text": "A claim.",
                "evidence_ids": ["S1"],
            }],
            "evidence": evidence,
        })


def test_fact_checker_rejects_duplicate_claim_ids(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    claim = {
        "claim_id": "C1",
        "claim_text": "A claim.",
        "evidence_ids": ["S1"],
    }

    with pytest.raises(FactCheckerError, match="Duplicate claim ID"):
        checker({
            "claims": [claim, claim],
            "evidence": evidence,
        })


def test_fact_checker_rejects_empty_claim_text(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Claim text"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": " ",
                "evidence_ids": ["S1"],
            }],
            "evidence": evidence,
        })


def test_fact_checker_rejects_claim_without_evidence_ids(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="must reference evidence"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": "A claim.",
                "evidence_ids": [],
            }],
            "evidence": evidence,
        })


def test_fact_checker_rejects_unknown_evidence_id(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="unknown evidence"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": "A claim.",
                "evidence_ids": ["S999"],
            }],
            "evidence": evidence,
        })


def test_fact_checker_rejects_duplicate_claim_evidence_ids(evidence):
    checker = FactCheckerNode(FakeSemanticVerifier())

    with pytest.raises(FactCheckerError, match="Duplicate evidence references"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": "A claim.",
                "evidence_ids": ["S1", "S1"],
            }],
            "evidence": evidence,
        })


def test_fact_checker_wraps_semantic_verifier_error(evidence):
    verifier = FakeSemanticVerifier(
        error=SemanticVerifierError("Verification failed")
    )
    checker = FactCheckerNode(verifier)

    with pytest.raises(FactCheckerError, match="Verification failed"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": "A claim.",
                "evidence_ids": ["S1"],
            }],
            "evidence": evidence,
        })


def test_fact_checker_wraps_invalid_verifier_input(evidence):
    verifier = FakeSemanticVerifier(
        error=ValueError("Invalid input")
    )
    checker = FactCheckerNode(verifier)

    with pytest.raises(FactCheckerError, match="Invalid verification input"):
        checker({
            "claims": [{
                "claim_id": "C1",
                "claim_text": "A claim.",
                "evidence_ids": ["S1"],
            }],
            "evidence": evidence,
        })
