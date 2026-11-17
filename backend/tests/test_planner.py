import pytest

from app.agents.planner import PlannerError, PlannerNode


class MockLLM:
    def __init__(self, response="1. First step\n2. Second step\n3. Third step"):
        self.response = response
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response


def test_planner_returns_research_plan():
    llm = MockLLM()
    planner = PlannerNode(llm=llm)

    result = planner({
        "question": "How does RAG improve LLM factual accuracy?"
    })

    assert result["research_plan"] == [
        "First step",
        "Second step",
        "Third step",
    ]


def test_planner_rejects_empty_question():
    planner = PlannerNode(llm=MockLLM())

    with pytest.raises(PlannerError, match="non-empty"):
        planner({"question": "   "})


def test_planner_rejects_empty_model_response():
    planner = PlannerNode(llm=MockLLM(response="   "))

    with pytest.raises(PlannerError, match="empty research plan"):
        planner({"question": "Explain RAG"})


def test_planner_passes_question_to_llm():
    llm = MockLLM()
    planner = PlannerNode(llm=llm)

    planner({"question": "Explain hybrid retrieval"})

    assert "Explain hybrid retrieval" in llm.last_prompt
    assert llm.last_system_prompt is not None


def test_planner_handles_markdown_bullets():
    llm = MockLLM(
        response="- Identify the research problem\n"
        "* Examine relevant literature\n"
        "3) Compare findings"
    )
    planner = PlannerNode(llm=llm)

    result = planner({"question": "Explain RAG"})

    assert result["research_plan"] == [
        "Identify the research problem",
        "Examine relevant literature",
        "Compare findings",
    ]


def test_planner_rejects_non_string_question():
    planner = PlannerNode(llm=MockLLM())

    with pytest.raises(PlannerError, match="non-empty string"):
        planner({"question": None})


def test_planner_rejects_non_string_model_response():
    planner = PlannerNode(llm=MockLLM(response=None))

    with pytest.raises(PlannerError, match="invalid response"):
        planner({"question": "Explain RAG"})
