import re

from app.llm import OllamaClient, OllamaError
from app.services.claim_verifier import (
    ClaimCitationError,
    extract_claims_from_answer,
    validate_claim_citations,
)
from app.services.search_service import SearchService
from app.services.semantic_verifier import (
    SemanticClaimVerifier,
    SemanticVerifierError,
)


class RAGError(Exception):
    """Raised when RAG answer generation or verification fails."""


class RAGService:
    INSUFFICIENT_EVIDENCE = (
        "I could not find sufficient evidence in the available documents "
        "to answer this question reliably."
    )

    SYSTEM_PROMPT = """
You are AURA, a strict evidence-grounded research assistant.

CRITICAL: CITATIONS ARE MANDATORY.

You must cite every factual sentence using the source IDs
provided in the evidence.

STRICT OUTPUT RULES:

1. Output factual bullet points only.
2. Every bullet must contain at least one complete sentence.
3. EVERY factual sentence must contain a citation.
4. Place the citation immediately before the final punctuation.
5. Never output a factual sentence without a citation.
6. Use only the supplied source IDs.
7. Never invent citations or evidence.
8. Do not write introductions, headings, or uncited summaries.
9. Do not output commentary outside the bullet list.

CORRECT FORMAT:
- Section-aware chunking improved retrieval accuracy
  from 71% to 84% [S1].

INCORRECT FORMAT:
- Section-aware chunking improved retrieval accuracy
  from 71% to 84%.

- Section-aware chunking improved retrieval accuracy
  from 71% to 84%. [S1]

If evidence is insufficient, return exactly:
INSUFFICIENT_EVIDENCE

A response containing even one uncited factual sentence
is INVALID.

Return only the answer.
"""

    def __init__(
        self,
        search_service=None,
        ollama_client=None,
        semantic_verifier=None,
    ):
        self.search_service = (
            search_service if search_service is not None else SearchService()
        )
        self.ollama_client = (
            ollama_client if ollama_client is not None else OllamaClient()
        )
        self.semantic_verifier = (
            semantic_verifier
            if semantic_verifier is not None
            else SemanticClaimVerifier(self.ollama_client)
        )

    def answer(self, workspace_id: int, question: str, limit: int = 5):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.search_service.search(
            workspace_id=workspace_id,
            query=question,
            limit=limit,
        )

        if not results:
            return {
                "question": question,
                "answer": self.INSUFFICIENT_EVIDENCE,
                "citations": [],
                "evidence_count": 0,
                "claims": [],
            }

        evidence_sections = []
        citation_map = {}
        evidence_text_map = {}

        for index, result in enumerate(results, start=1):
            citation_id = f"S{index}"

            citation_map[citation_id] = {
                "source_id": citation_id,
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_id": result["chunk_id"],
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "score": result["score"],
            }

            evidence_text_map[citation_id] = result["text"]

            evidence_sections.append(
                f"""
[{citation_id}]
Document: {result['filename']}
Pages: {result['page_start']}-{result['page_end']}
Evidence:
{result['text']}
"""
            )

        evidence = "\n".join(evidence_sections)

        valid_citation_ids = ", ".join(
            f"[S{index}]"
            for index in range(1, len(results) + 1)
        )

        prompt = f"""
Research question:
{question}

VALID CITATION IDS:
{valid_citation_ids}

AVAILABLE EVIDENCE:
{evidence}

TASK:
Answer the research question using the available evidence.

CITATION REQUIREMENTS — HIGHEST PRIORITY:

1. Every factual sentence MUST contain a citation.
2. Every bullet MUST contain at least one citation.
3. Citations MUST appear before sentence-ending punctuation.
4. Use only these valid IDs: {valid_citation_ids}
5. Match every claim to the evidence that supports it.
6. Never produce an uncited factual statement.

REQUIRED FORMAT EXAMPLE:

- A research finding supported by the first evidence source [S1].
- Another finding supported by the second evidence source [S2].

CONTENT TO COVER WHEN SUPPORTED:

- Major research findings.
- Quantitative results and comparisons.
- Conclusions and recommendations.
- Limitations and future work.

Do not include introductions, headings, or summary paragraphs.

If the evidence cannot adequately answer the question,
return exactly:
INSUFFICIENT_EVIDENCE

Return only cited factual bullet points.
"""

        try:
            generated_answer = self.ollama_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
        except OllamaError as exc:
            raise RAGError(f"Answer generation failed: {exc}") from exc

        generated_answer = generated_answer.strip()

        fallback_pattern = (
            r"(?im)^[ \t]*INSUFFICIENT_EVIDENCE[ \t]*[.!]?[ \t]*$"
        )

        fallback_match = re.search(
            fallback_pattern,
            generated_answer,
        )

        if fallback_match:
            answer_prefix = generated_answer[:fallback_match.start()].strip()

            if not answer_prefix:
                return {
                    "question": question,
                    "answer": self.INSUFFICIENT_EVIDENCE,
                    "citations": [],
                    "evidence_count": len(results),
                    "claims": [],
                }

            # Discard the fallback marker and everything after it.
            generated_answer = answer_prefix

        elif "INSUFFICIENT_EVIDENCE" in generated_answer.upper():
            # Reject inline or malformed fallback markers.
            return {
                "question": question,
                "answer": self.INSUFFICIENT_EVIDENCE,
                "citations": [],
                "evidence_count": len(results),
                "claims": [],
            }

        try:
            claims = extract_claims_from_answer(generated_answer)

            validated_claims = validate_claim_citations(
                claims=claims,
                evidence_map=citation_map,
            )

        except ClaimCitationError:
            return {
                "question": question,
                "answer": self.INSUFFICIENT_EVIDENCE,
                "citations": [],
                "evidence_count": len(results),
                "claims": [],
            }

        # Semantically verify each claim against only its cited evidence.
        verification_results = []

        try:
            for claim in validated_claims:
                claim_evidence = [
                    {
                        "citation_id": citation["citation_id"],
                        "text": evidence_text_map[citation["citation_id"]],
                    }
                    for citation in claim["citations"]
                ]

                verification_claim_text = re.sub(
                    r"\s*\[S\d+\]",
                    "",
                    claim["text"],
                ).strip()

                verification = self.semantic_verifier.verify_claim(
                    claim_id=claim["claim_id"],
                    claim_text=verification_claim_text,
                    evidence=claim_evidence,
                )

                verification_results.append({
                    "claim_id": verification.claim_id,
                    "claim_text": claim["text"],
                    "status": verification.status.value,
                    "rationale": verification.rationale,
                    "evidence_ids": verification.evidence_ids,
                })

        except SemanticVerifierError as exc:
            raise RAGError(
                "Semantic claim verification failed."
            ) from exc

        # Only semantically supported claims may appear in the answer.
        supported_claims = [
            claim
            for claim, verification in zip(
                validated_claims,
                verification_results,
            )
            if verification["status"] == "supported"
        ]

        supported_answer = "\n".join(
            claim["text"] for claim in supported_claims
        )

        # Collect citations only from claims that passed verification.
        unique_cited_ids = list(dict.fromkeys(
            citation["citation_id"]
            for claim in supported_claims
            for citation in claim["citations"]
        ))

        valid_citations = [
            citation_map[citation_id]
            for citation_id in unique_cited_ids
        ]

        return {
            "question": question,
            "answer": (
                supported_answer
                if supported_answer
                else self.INSUFFICIENT_EVIDENCE
            ),
            "citations": valid_citations,
            "evidence_count": len(results),
            # Preserve every verification result for transparency.
            "claims": verification_results,
        }
