import pytest

from app.agents.researcher import ResearcherError, ResearcherNode


class MockSearchService:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def make_evidence(document_id=1, chunk_id=10):
    return {
        "document_id": document_id,
        "filename": "research.pdf",
        "chunk_id": chunk_id,
        "text": "Retrieved research evidence.",
        "page_start": 1,
        "page_end": 2,
        "score": 0.85,
    }


def test_researcher_retrieves_evidence_for_each_step():
    search = MockSearchService([make_evidence()])
    researcher = ResearcherNode(search_service=search)

    result = researcher({
        "workspace_id": 7,
        "research_plan": ["Dense retrieval", "Sparse retrieval"],
    })

    assert len(search.calls) == 2
    assert all(call["workspace_id"] == 7 for call in search.calls)
    assert len(result["evidence"]) == 1


def test_researcher_deduplicates_evidence():
    search = MockSearchService([make_evidence()])
    researcher = ResearcherNode(search_service=search)

    result = researcher({
        "workspace_id": 7,
        "research_plan": ["Step one", "Step two"],
    })

    assert len(result["evidence"]) == 1


def test_researcher_preserves_evidence_metadata():
    search = MockSearchService([make_evidence()])
    researcher = ResearcherNode(search_service=search)

    result = researcher({
        "workspace_id": 7,
        "research_plan": ["Dense retrieval"],
    })

    evidence = result["evidence"][0]

    assert evidence["document_id"] == 1
    assert evidence["chunk_id"] == 10
    assert evidence["filename"] == "research.pdf"
    assert evidence["page_start"] == 1
    assert evidence["page_end"] == 2
    assert evidence["research_query"] == "Dense retrieval"


def test_researcher_rejects_invalid_workspace():
    researcher = ResearcherNode(search_service=MockSearchService())

    with pytest.raises(ResearcherError, match="workspace ID"):
        researcher({
            "workspace_id": 0,
            "research_plan": ["Dense retrieval"],
        })


def test_researcher_rejects_empty_plan():
    researcher = ResearcherNode(search_service=MockSearchService())

    with pytest.raises(ResearcherError, match="cannot be empty"):
        researcher({
            "workspace_id": 7,
            "research_plan": [],
        })


def test_researcher_wraps_search_errors():
    class FailingSearchService:
        def search(self, **kwargs):
            raise RuntimeError("Database unavailable")

    researcher = ResearcherNode(
        search_service=FailingSearchService()
    )

    with pytest.raises(ResearcherError, match="retrieval failed"):
        researcher({
            "workspace_id": 7,
            "research_plan": ["Dense retrieval"],
        })


def test_researcher_rejects_boolean_workspace_id():
    researcher = ResearcherNode(search_service=MockSearchService())

    with pytest.raises(ResearcherError, match="workspace ID"):
        researcher({
            "workspace_id": True,
            "research_plan": ["Dense retrieval"],
        })


def test_researcher_rejects_malformed_evidence():
    search = MockSearchService([{"text": "Evidence without IDs"}])
    researcher = ResearcherNode(search_service=search)

    with pytest.raises(ResearcherError, match="evidence metadata"):
        researcher({
            "workspace_id": 7,
            "research_plan": ["Dense retrieval"],
        })


def test_researcher_rejects_non_list_research_plan():
    researcher = ResearcherNode(search_service=MockSearchService())

    with pytest.raises(ResearcherError, match="Research plan"):
        researcher({
            "workspace_id": 7,
            "research_plan": "Dense retrieval",
        })


def test_researcher_rejects_non_list_search_results():
    search = MockSearchService(results="invalid response")
    researcher = ResearcherNode(search_service=search)

    with pytest.raises(ResearcherError, match="Invalid retrieval response"):
        researcher({
            "workspace_id": 7,
            "research_plan": ["Dense retrieval"],
        })


@pytest.mark.parametrize("invalid_result", [None, "invalid", 123])
def test_researcher_rejects_non_dictionary_evidence(invalid_result):
    search = MockSearchService(results=[invalid_result])
    researcher = ResearcherNode(search_service=search)

    with pytest.raises(ResearcherError, match="evidence metadata"):
        researcher({
            "workspace_id": 7,
            "research_plan": ["Dense retrieval"],
        })


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", None),
        ("document_id", True),
        ("document_id", 0),
        ("chunk_id", None),
        ("chunk_id", True),
        ("chunk_id", 0),
    ],
)
def test_researcher_rejects_invalid_evidence_identifiers(field, value):
    evidence = make_evidence()
    evidence[field] = value

    search = MockSearchService(results=[evidence])
    researcher = ResearcherNode(search_service=search)

    with pytest.raises(ResearcherError, match="evidence metadata"):
        researcher({
            "workspace_id": 7,
            "research_plan": ["Dense retrieval"],
        })
