import json

import pytest

from app.llm import OllamaError
from app.services.semantic_verifier import (
    SemanticClaimVerifier,
    SemanticVerifierError,
    VerificationStatus,
)


class FakeOllamaClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def evidence():
    return [
        {
            "citation_id": "S1",
            "text": "The study reported a 15% increase in retrieval accuracy.",
        },
        {
            "citation_id": "S2",
            "text": "The experiment used 500 research documents.",
        },
    ]


@pytest.mark.parametrize(
    "status",
    [
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
    ],
)
def test_verify_claim_accepts_all_valid_statuses(status, evidence):
    response = {
        "status": status,
        "rationale": "Assessment based on the supplied evidence.",
        "evidence_ids": ["S1"],
    }

    client = FakeOllamaClient(json.dumps(response))
    verifier = SemanticClaimVerifier(client)

    result = verifier.verify_claim(
        claim_id="C1",
        claim_text="The study improved retrieval accuracy.",
        evidence=evidence,
    )

    assert result.claim_id == "C1"
    assert result.status == VerificationStatus(status)
    assert result.rationale
    assert result.evidence_ids == ["S1"]


def test_verify_claim_rejects_empty_claim():
    verifier = SemanticClaimVerifier(FakeOllamaClient())

    with pytest.raises(ValueError, match="Claim text"):
        verifier.verify_claim("C1", " ", [])


def test_verify_claim_rejects_empty_claim_id():
    verifier = SemanticClaimVerifier(FakeOllamaClient())

    with pytest.raises(ValueError, match="Claim ID"):
        verifier.verify_claim(" ", "A research claim.", [])


def test_verify_claim_returns_insufficient_without_evidence():
    client = FakeOllamaClient()
    verifier = SemanticClaimVerifier(client)

    result = verifier.verify_claim(
        claim_id="C1",
        claim_text="The study improved retrieval accuracy.",
        evidence=[],
    )

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence_ids == []
    assert client.calls == []


def test_verify_claim_rejects_malformed_json(evidence):
    client = FakeOllamaClient("This is not JSON.")
    verifier = SemanticClaimVerifier(client)

    with pytest.raises(SemanticVerifierError, match="invalid claim verification"):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_rejects_invalid_status(evidence):
    response = {
        "status": "probably_true",
        "rationale": "The evidence seems relevant.",
        "evidence_ids": ["S1"],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    with pytest.raises(SemanticVerifierError):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_rejects_empty_rationale(evidence):
    response = {
        "status": "supported",
        "rationale": "   ",
        "evidence_ids": ["S1"],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    with pytest.raises(SemanticVerifierError):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_rejects_unknown_evidence_id(evidence):
    response = {
        "status": "supported",
        "rationale": "The evidence supports the claim.",
        "evidence_ids": ["S999"],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    with pytest.raises(SemanticVerifierError, match="evidence"):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_rejects_extra_response_fields(evidence):
    response = {
        "status": "supported",
        "rationale": "The evidence supports the claim.",
        "evidence_ids": ["S1"],
        "document_id": 999,
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    with pytest.raises(SemanticVerifierError):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_translates_ollama_error(evidence):
    client = FakeOllamaClient(error=OllamaError("Connection failed"))
    verifier = SemanticClaimVerifier(client)

    with pytest.raises(SemanticVerifierError, match="could not complete"):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_uses_zero_temperature(evidence):
    response = {
        "status": "supported",
        "rationale": "The evidence supports the claim.",
        "evidence_ids": ["S1"],
    }

    client = FakeOllamaClient(json.dumps(response))
    verifier = SemanticClaimVerifier(client)

    verifier.verify_claim("C1", "A research claim.", evidence)

    assert client.calls[0]["temperature"] == 0.0


def test_verify_claim_deduplicates_evidence_ids(evidence):
    response = {
        "status": "supported",
        "rationale": "Both references support the claim.",
        "evidence_ids": ["S1", "S1", "S2"],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    result = verifier.verify_claim("C1", "A research claim.", evidence)

    assert result.evidence_ids == ["S1", "S2"]


def test_verify_claim_rejects_duplicate_input_evidence_ids():
    evidence = [
        {"citation_id": "S1", "text": "First source."},
        {"citation_id": "S1", "text": "Second source."},
    ]

    verifier = SemanticClaimVerifier(FakeOllamaClient())

    with pytest.raises(ValueError, match="Duplicate"):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_rejects_supported_status_without_evidence_ids(evidence):
    response = {
        "status": "supported",
        "rationale": "The evidence supports the claim.",
        "evidence_ids": [],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    with pytest.raises(SemanticVerifierError, match="did not identify evidence"):
        verifier.verify_claim("C1", "A research claim.", evidence)


def test_verify_claim_allows_insufficient_status_without_evidence_ids(evidence):
    response = {
        "status": "insufficient_evidence",
        "rationale": "The supplied evidence does not establish the claim.",
        "evidence_ids": [],
    }

    verifier = SemanticClaimVerifier(FakeOllamaClient(json.dumps(response)))

    result = verifier.verify_claim("C1", "A research claim.", evidence)

    assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence_ids == []


def test_verification_result_rejects_empty_rationale():
    from app.services.semantic_verifier import ClaimVerificationResult

    with pytest.raises(ValueError):
        ClaimVerificationResult(
            claim_id="C1",
            status=VerificationStatus.SUPPORTED,
            rationale=" ",
            evidence_ids=["S1"],
        )