from app.agents.state import ResearchState
from app.services.semantic_verifier import (
    SemanticClaimVerifier,
    SemanticVerifierError,
)


class FactCheckerError(Exception):
    """Raised when research claim verification fails."""


class FactCheckerNode:
    """Verifies Analyst claims against registered research evidence."""

    def __init__(
        self,
        verifier: SemanticClaimVerifier | None = None,
    ):
        self.verifier = (
            verifier
            if verifier is not None
            else SemanticClaimVerifier()
        )

    def __call__(self, state: ResearchState) -> dict:
        claims = state.get("claims", [])
        evidence = state.get("evidence", [])

        if not isinstance(claims, list):
            raise FactCheckerError("Claims must be a list.")

        if not isinstance(evidence, list):
            raise FactCheckerError("Evidence must be a list.")

        evidence_by_id = {}

        for item in evidence:
            if not isinstance(item, dict):
                raise FactCheckerError(
                    "Each evidence item must be a dictionary."
                )

            citation_id = item.get("citation_id")
            text = item.get("text")

            if not isinstance(citation_id, str) or not citation_id.strip():
                raise FactCheckerError("Invalid evidence citation ID.")

            if citation_id in evidence_by_id:
                raise FactCheckerError(
                    f"Duplicate evidence ID: {citation_id}"
                )

            if not isinstance(text, str) or not text.strip():
                raise FactCheckerError(
                    f"Evidence text is empty: {citation_id}"
                )

            evidence_by_id[citation_id] = text.strip()

        verification_results = []
        seen_claim_ids = set()

        for claim in claims:
            if not isinstance(claim, dict):
                raise FactCheckerError(
                    "Each claim must be a dictionary."
                )

            claim_id = claim.get("claim_id")
            claim_text = claim.get("claim_text")
            evidence_ids = claim.get("evidence_ids")

            if not isinstance(claim_id, str) or not claim_id.strip():
                raise FactCheckerError("Claim ID cannot be empty.")

            if claim_id in seen_claim_ids:
                raise FactCheckerError(
                    f"Duplicate claim ID: {claim_id}"
                )

            seen_claim_ids.add(claim_id)

            if not isinstance(claim_text, str) or not claim_text.strip():
                raise FactCheckerError(
                    f"Claim text cannot be empty: {claim_id}"
                )

            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise FactCheckerError(
                    f"Claim {claim_id} must reference evidence."
                )

            if any(
                not isinstance(evidence_id, str)
                or not evidence_id.strip()
                for evidence_id in evidence_ids
            ):
                raise FactCheckerError(
                    f"Invalid evidence references in claim {claim_id}."
                )

            if len(evidence_ids) != len(set(evidence_ids)):
                raise FactCheckerError(
                    f"Duplicate evidence references in claim {claim_id}."
                )

            unknown_ids = set(evidence_ids) - set(evidence_by_id)

            if unknown_ids:
                raise FactCheckerError(
                    f"Claim {claim_id} references unknown evidence: "
                    f"{sorted(unknown_ids)}"
                )

            claim_evidence = [
                {
                    "citation_id": evidence_id,
                    "text": evidence_by_id[evidence_id],
                }
                for evidence_id in evidence_ids
            ]

            try:
                result = self.verifier.verify_claim(
                    claim_id=claim_id,
                    claim_text=claim_text.strip(),
                    evidence=claim_evidence,
                )
            except SemanticVerifierError as exc:
                raise FactCheckerError(
                    f"Verification failed for claim {claim_id}."
                ) from exc
            except ValueError as exc:
                raise FactCheckerError(
                    f"Invalid verification input for claim {claim_id}."
                ) from exc

            verification_results.append(result.model_dump(mode="json"))

        return {"verification_results": verification_results}
