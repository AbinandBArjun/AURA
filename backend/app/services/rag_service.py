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
You are AURA, an evidence-grounded research assistant.

Rules:
1. Answer using only the supplied evidence.
2. Never invent facts, statistics, page numbers, or sources.
3. Cite every substantive factual claim using the supplied citation IDs,
   for example [S1] or [S2].
4. If the evidence does not adequately answer the question, say:
   "INSUFFICIENT_EVIDENCE"
5. If sources contradict each other, explicitly identify the disagreement.
6. Be precise and distinguish evidence from interpretation.
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

        prompt = f"""
Research question:
{question}

Available evidence:
{evidence}

Write an evidence-based answer with citations.
"""

        try:
            generated_answer = self.ollama_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
        except OllamaError as exc:
            raise RAGError(f"Answer generation failed: {exc}") from exc

        if "INSUFFICIENT_EVIDENCE" in generated_answer.upper():
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

        # Collect validated citations in their original order.
        unique_cited_ids = list(dict.fromkeys(
            citation["citation_id"]
            for claim in validated_claims
            for citation in claim["citations"]
        ))

        valid_citations = [
            citation_map[citation_id]
            for citation_id in unique_cited_ids
        ]

        return {
            "question": question,
            "answer": generated_answer,
            "citations": valid_citations,
            "evidence_count": len(results),
            "claims": verification_results,
        }
