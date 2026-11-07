import re

from app.llm import OllamaClient, OllamaError
from app.services.search_service import SearchService


class RAGError(Exception):
    """Raised when RAG answer generation fails."""


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

    def __init__(self, search_service=None, ollama_client=None):
        self.search_service = search_service or SearchService()
        self.ollama_client = ollama_client or OllamaClient()

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
            }

        evidence_sections = []
        citation_map = {}

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
            }

        # Extract every citation ID used by the model.
        cited_ids = re.findall(r"\[(S\d+)\]", generated_answer)

        # Reject answers containing citation IDs that were never supplied.
        invalid_ids = {
            citation_id
            for citation_id in cited_ids
            if citation_id not in citation_map
        }

        if invalid_ids:
            return {
                "question": question,
                "answer": self.INSUFFICIENT_EVIDENCE,
                "citations": [],
                "evidence_count": len(results),
            }

        # Preserve citation order while removing duplicates.
        unique_cited_ids = list(dict.fromkeys(cited_ids))

        valid_citations = [
            citation_map[citation_id]
            for citation_id in unique_cited_ids
        ]

        # Reject answers without valid citations.
        if not valid_citations:
            return {
                "question": question,
                "answer": self.INSUFFICIENT_EVIDENCE,
                "citations": [],
                "evidence_count": len(results),
            }

        return {
            "question": question,
            "answer": generated_answer,
            "citations": valid_citations,
            "evidence_count": len(results),
        }
