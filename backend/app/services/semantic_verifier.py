import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm import OllamaClient, OllamaError


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class _LLMVerificationResponse(BaseModel):
    """Strict schema for the LLM-generated verification response."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: VerificationStatus
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    """Validated semantic verification result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    status: VerificationStatus
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @classmethod
    def validate_evidence_consistency(
        cls,
        status: VerificationStatus,
        evidence_ids: list[str],
    ) -> None:
        if (
            status != VerificationStatus.INSUFFICIENT_EVIDENCE
            and not evidence_ids
        ):
            raise ValueError(
                "Evidence IDs are required for this verification status."
            )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Duplicate evidence IDs are not allowed.")


class SemanticVerifierError(Exception):
    """Raised when semantic verification cannot be completed."""


class SemanticClaimVerifier:
    SYSTEM_PROMPT = """
You are a research claim verification system.

Evaluate the supplied claim using ONLY the supplied evidence.

Do not use outside knowledge.
Do not assume that a claim is true merely because it sounds plausible.
Distinguish direct support from partial support.
Identify explicit contradictions when evidence conflicts with the claim.
If the evidence does not establish whether the claim is true, classify it
as insufficient_evidence.

Return ONLY valid JSON with exactly these fields:
{
  "status": "supported | partially_supported | contradicted | insufficient_evidence",
  "rationale": "A concise explanation based only on the evidence.",
  "evidence_ids": ["S1"]
}

Rules:
- Use only evidence IDs present in the input.
- Do not invent evidence IDs.
- evidence_ids must identify the evidence relevant to your assessment.
- Do not include document metadata or additional fields.
- Do not wrap the JSON in Markdown code fences.
"""

    def __init__(self, ollama_client: OllamaClient | None = None):
        self.ollama_client = ollama_client or OllamaClient()

    def verify_claim(
        self,
        claim_id: str,
        claim_text: str,
        evidence: list[dict[str, Any]],
    ) -> ClaimVerificationResult:
        if not claim_id or not claim_id.strip():
            raise ValueError("Claim ID cannot be empty.")

        if not claim_text or not claim_text.strip():
            raise ValueError("Claim text cannot be empty.")

        if not isinstance(evidence, list):
            raise ValueError("Evidence must be a list.")

        evidence_by_id: dict[str, str] = {}

        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("Each evidence item must be a dictionary.")

            evidence_id = item.get("citation_id")
            evidence_text = item.get("text")

            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError("Evidence citation ID cannot be empty.")

            if not isinstance(evidence_text, str) or not evidence_text.strip():
                raise ValueError("Evidence text cannot be empty.")

            if evidence_id in evidence_by_id:
                raise ValueError("Duplicate evidence citation ID.")

            evidence_by_id[evidence_id] = evidence_text

        if not evidence_by_id:
            return ClaimVerificationResult(
                claim_id=claim_id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                rationale="No evidence was supplied for this claim.",
                evidence_ids=[],
            )

        prompt_data = {
            "claim": claim_text.strip(),
            "evidence": [
                {
                    "citation_id": evidence_id,
                    "text": evidence_text,
                }
                for evidence_id, evidence_text in evidence_by_id.items()
            ],
        }

        prompt = (
            "Verify the following research claim against the supplied evidence.\n\n"
            + json.dumps(prompt_data, ensure_ascii=False, indent=2)
        )

        try:
            raw_response = self.ollama_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
            )
        except OllamaError as exc:
            raise SemanticVerifierError(
                "The LLM could not complete claim verification."
            ) from exc

        try:
            parsed_response = json.loads(raw_response)
            validated_response = _LLMVerificationResponse.model_validate(
                parsed_response
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise SemanticVerifierError(
                "The LLM returned an invalid claim verification response."
            ) from exc

        unknown_evidence_ids = (
            set(validated_response.evidence_ids) - set(evidence_by_id)
        )

        if unknown_evidence_ids:
            raise SemanticVerifierError(
                "The LLM referenced evidence that was not supplied."
            )

        if (
            validated_response.status
            != VerificationStatus.INSUFFICIENT_EVIDENCE
            and not validated_response.evidence_ids
        ):
            raise SemanticVerifierError(
                "The LLM did not identify evidence for its assessment."
            )

        return ClaimVerificationResult(
            claim_id=claim_id,
            status=validated_response.status,
            rationale=validated_response.rationale.strip(),
            evidence_ids=list(dict.fromkeys(validated_response.evidence_ids)),
        )
