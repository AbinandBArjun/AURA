import pytest

from app.llm import OllamaError
from app.services.claim_verifier import (
    extract_claims_from_answer,
)
from app.services.rag_service import RAGService, RAGError


class FakeSearchService:
    def __init__(self, results=None):
        self.results = results or []

    def search(self, workspace_id, query, limit):
        return self.results


class FakeOllamaClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_prompt = None

    def generate(self, prompt, system_prompt, temperature):
        self.last_prompt = prompt

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def sample_results():
    return [
        {
            "document_id": 1,
            "filename": "research_paper.pdf",
            "chunk_id": 10,
            "page_start": 2,
            "page_end": 3,
            "text": "RAG combines retrieval with language generation.",
            "score": 0.85,
        },
        {
            "document_id": 2,
            "filename": "technical_report.pdf",
            "chunk_id": 20,
            "page_start": 5,
            "page_end": 5,
            "text": "Retrieved evidence can improve answer relevance.",
            "score": 0.78,
        },
    ]


def test_successful_answer_includes_valid_citations(sample_results):
    search = FakeSearchService(sample_results)
    ollama = FakeOllamaClient(
        response="RAG combines retrieval and generation [S1]."
    )

    service = RAGService(search, ollama)

    result = service.answer(
        workspace_id=1,
        question="What is RAG?",
    )

    assert "[S1]" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["filename"] == "research_paper.pdf"
    assert result["citations"][0]["page_start"] == 2
    assert result["evidence_count"] == 2
    assert "[S1]" in ollama.last_prompt
    assert "[S2]" in ollama.last_prompt


def test_no_search_results_returns_insufficient_evidence():
    service = RAGService(
        FakeSearchService([]),
        FakeOllamaClient(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["evidence_count"] == 0


def test_model_insufficient_evidence_response(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="INSUFFICIENT_EVIDENCE"),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []


def test_uncited_model_response_is_rejected(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation."),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []


def test_ollama_failure_raises_rag_error(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(error=OllamaError("Connection failed")),
    )

    with pytest.raises(RAGError):
        service.answer(1, "What is RAG?")


def test_rag_rejects_uncited_sentence(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response=(
                "RAG combines retrieval and generation [S1]. "
                "It always guarantees accurate answers."
            )
        ),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []


def test_rag_accepts_multiple_cited_sentences(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response=(
                "RAG combines retrieval and generation [S1]. "
                "Retrieved evidence can improve answer relevance [S2]."
            )
        ),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] != RAGService.INSUFFICIENT_EVIDENCE
    assert len(result["citations"]) == 2


def test_rag_rejects_fabricated_citation_in_sentence(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response="RAG guarantees perfect accuracy [S99]."
        ),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []


def test_extract_claims_from_answer():
    answer = (
        "RAG combines retrieval and generation [S1]. "
        "Evidence improves relevance [S2]."
    )

    claims = extract_claims_from_answer(answer)

    assert len(claims) == 2
    assert claims[0]["claim_id"] == "C1"
    assert claims[0]["citation_ids"] == ["S1"]
    assert claims[1]["claim_id"] == "C2"
    assert claims[1]["citation_ids"] == ["S2"]
