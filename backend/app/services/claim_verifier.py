
import re


class ClaimCitationError(ValueError):
    """Raised when claim or citation validation fails."""


REQUIRED_SOURCE_FIELDS = {
    "document_id",
    "filename",
    "chunk_id",
    "page_start",
    "page_end",
    "score",
}


def validate_claim_citations(
    claims: list[dict],
    evidence_map: dict[str, dict],
) -> list[dict]:
    """Validate claim citations against server-generated evidence."""

    if not isinstance(claims, list):
        raise ClaimCitationError("Claims must be a list.")

    if not isinstance(evidence_map, dict):
        raise ClaimCitationError("Evidence map must be a dictionary.")

    validated_claims = []
    seen_claim_ids = set()

    for claim in claims:
        if not isinstance(claim, dict):
            raise ClaimCitationError("Each claim must be an object.")

        claim_id = claim.get("claim_id")
        text = claim.get("text")
        citation_ids = claim.get("citation_ids")

        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ClaimCitationError("Claim ID cannot be empty.")

        if claim_id in seen_claim_ids:
            raise ClaimCitationError(
                f"Duplicate claim ID: {claim_id}"
            )

        seen_claim_ids.add(claim_id)

        if not isinstance(text, str) or not text.strip():
            raise ClaimCitationError("Claim text cannot be empty.")

        if not isinstance(citation_ids, list) or not citation_ids:
            raise ClaimCitationError(
                f"Claim {claim_id} must contain citations."
            )

        if not all(isinstance(item, str) for item in citation_ids):
            raise ClaimCitationError("Citation IDs must be strings.")

        unique_ids = list(dict.fromkeys(citation_ids))
        validated_citations = []

        for citation_id in unique_ids:
            if not isinstance(citation_id, str):
                raise ClaimCitationError("Citation IDs must be strings.")

            if not re.fullmatch(r"S[1-9]\d*", citation_id):
                raise ClaimCitationError(
                    f"Invalid citation ID format: {citation_id}"
                )

            if citation_id not in evidence_map:
                raise ClaimCitationError(
                    f"Unknown citation ID: {citation_id}"
                )

            source = evidence_map[citation_id]

            if not isinstance(source, dict):
                raise ClaimCitationError(
                    f"Invalid evidence metadata: {citation_id}"
                )

            missing_fields = REQUIRED_SOURCE_FIELDS - source.keys()

            if missing_fields:
                raise ClaimCitationError(
                    f"Incomplete evidence metadata for {citation_id}: "
                    f"{sorted(missing_fields)}"
                )

            validated_citations.append({
                "citation_id": citation_id,
                "source": dict(source),
            })

        validated_claims.append({
            "claim_id": claim_id,
            "text": text.strip(),
            "citations": validated_citations,
        })

    return validated_claims


def extract_claims_from_answer(answer: str) -> list[dict]:
    """
    Convert answer sentences into claim objects with citation IDs.

    Supports Markdown numbered and bulleted lists while retaining
    strict sentence-level citation validation.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ClaimCitationError("Answer cannot be empty.")

    # Remove Markdown list markers at the beginning of lines.
    # This prevents numbered-list periods from becoming claims.
    normalized_answer = re.sub(
        r"(?m)^[ \t]*(?:\d+[.)]|[-*+])[ \t]+",
        "",
        answer.strip(),
    )

    # Split individual sentences, not entire list items.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized_answer,
    )

    claims = []

    for index, sentence in enumerate(sentences, start=1):
        sentence = sentence.strip()

        if not sentence:
            continue

        citation_ids = re.findall(
            r"\[(S\d+)\]",
            sentence,
        )

        claims.append({
            "claim_id": f"C{index}",
            "text": sentence,
            "citation_ids": citation_ids,
        })

    if not claims:
        raise ClaimCitationError("No claims found in answer.")

    return claims
