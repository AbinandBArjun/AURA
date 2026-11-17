import pytest

from app.llm import OllamaError
from app.services.claim_verifier import extract_claims_from_answer
from app.services.rag_service import RAGService, RAGError
from app.services.semantic_verifier import (
    ClaimVerificationResult,
    SemanticVerifierError,
    VerificationStatus,
)


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


class FakeSemanticVerifier:
    def __init__(self, status="supported", error=None):
        self.status = VerificationStatus(status)
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
            evidence_ids=[item["citation_id"] for item in evidence],
        )


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
    verifier = FakeSemanticVerifier()

    service = RAGService(search, ollama, verifier)

    result = service.answer(1, "What is RAG?")

    assert "[S1]" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["filename"] == "research_paper.pdf"
    assert result["citations"][0]["page_start"] == 2
    assert result["evidence_count"] == 2
    assert result["claims"][0]["status"] == "supported"
    assert len(verifier.calls) == 1


def test_no_search_results_returns_insufficient_evidence():
    service = RAGService(
        FakeSearchService([]),
        FakeOllamaClient(),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["evidence_count"] == 0
    assert result["claims"] == []


def test_model_insufficient_evidence_response(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="INSUFFICIENT_EVIDENCE"),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["claims"] == []


def test_uncited_model_response_is_rejected(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation."),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["claims"] == []


def test_ollama_failure_raises_rag_error(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(error=OllamaError("Connection failed")),
        FakeSemanticVerifier(),
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
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["claims"] == []


def test_rag_accepts_multiple_cited_sentences(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response=(
                "RAG combines retrieval and generation [S1]. "
                "Retrieved evidence can improve answer relevance [S2]."
            )
        ),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] != RAGService.INSUFFICIENT_EVIDENCE
    assert len(result["citations"]) == 2
    assert len(result["claims"]) == 2


def test_rag_rejects_fabricated_citation_in_sentence(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG guarantees perfect accuracy [S99]."),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == RAGService.INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert result["claims"] == []


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


@pytest.mark.parametrize(
    "status",
    [
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
    ],
)
def test_rag_exposes_all_semantic_statuses(sample_results, status):
    verifier = FakeSemanticVerifier(status=status)

    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation [S1]."),
        verifier,
    )

    result = service.answer(1, "What is RAG?")

    assert result["claims"][0]["status"] == status
    assert result["claims"][0]["claim_id"] == "C1"
    assert result["claims"][0]["evidence_ids"] == ["S1"]


def test_rag_verifier_receives_only_claim_citations(sample_results):
    verifier = FakeSemanticVerifier()

    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation [S1]."),
        verifier,
    )

    service.answer(1, "What is RAG?")

    assert verifier.calls[0]["evidence"] == [
        {
            "citation_id": "S1",
            "text": "RAG combines retrieval with language generation.",
        }
    ]


def test_rag_preserves_answer_when_claim_is_contradicted(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation [S1]."),
        FakeSemanticVerifier(status="contradicted"),
    )

    result = service.answer(1, "What is RAG?")

    assert result["answer"] == "RAG combines retrieval and generation [S1]."
    assert result["claims"][0]["status"] == "contradicted"


def test_rag_translates_semantic_verifier_error(sample_results):
    verifier = FakeSemanticVerifier(
        error=SemanticVerifierError("Invalid model response")
    )

    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation [S1]."),
        verifier,
    )

    with pytest.raises(RAGError, match="Semantic claim verification failed"):
        service.answer(1, "What is RAG?")


def test_rag_response_contains_claim_text_and_rationale(sample_results):
    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(response="RAG combines retrieval and generation [S1]."),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")

    claim = result["claims"][0]

    assert claim["claim_text"] == "RAG combines retrieval and generation [S1]."
    assert claim["rationale"] == "Assessment based on supplied evidence."


def test_semantic_verifier_receives_claim_without_citation_marker(sample_results):
    verifier = FakeSemanticVerifier()

    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response="RAG combines retrieval and generation [S1]."
        ),
        verifier,
    )

    result = service.answer(1, "What is RAG?")

    assert verifier.calls[0]["claim_text"] == (
        "RAG combines retrieval and generation."
    )

    assert result["claims"][0]["claim_text"] == (
        "RAG combines retrieval and generation [S1]."
    )


def test_rag_response_schema_accepts_semantic_claims(sample_results):
    from app.schemas.rag import RAGResponse

    service = RAGService(
        FakeSearchService(sample_results),
        FakeOllamaClient(
            response="RAG combines retrieval and generation [S1]."
        ),
        FakeSemanticVerifier(),
    )

    result = service.answer(1, "What is RAG?")
    response = RAGResponse.model_validate(result)

    assert response.claims[0].status == "supported"
    assert response.claims[0].claim_id == "C1"
    assert response.model_dump()["claims"][0]["evidence_ids"] == ["S1"]