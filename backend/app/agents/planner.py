import re

from app.agents.state import ResearchState
from app.llm import OllamaClient


class PlannerError(Exception):
    """Raised when research planning fails."""


class PlannerNode:
    """Generates a structured research plan from a user question."""

    def __init__(self, llm: OllamaClient | None = None):
        self.llm = llm if llm is not None else OllamaClient()

    def __call__(self, state: ResearchState) -> dict:
        question = state.get("question", "")

        if not isinstance(question, str) or not question.strip():
            raise PlannerError("Research question must be a non-empty string.")

        question = question.strip()

        system_prompt = (
            "You are AURA's research planning agent. "
            "Break the user's research question into clear, "
            "independent research steps. "
            "Return only a numbered list of 3 to 6 actionable steps. "
            "Do not answer the research question."
        )

        prompt = f"Research question:\n{question}"

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )
        except Exception as exc:
            raise PlannerError(
                f"Failed to generate research plan: {exc}"
            ) from exc

        if not isinstance(response, str):
            raise PlannerError("The planning model returned an invalid response.")

        research_plan = []

        for line in response.splitlines():
            step = line.strip()

            if not step:
                continue

            # Remove common numbered-list and bullet prefixes.
            step = re.sub(r"^(?:\d+[.)]\s*|[-*•]\s*)", "", step).strip()

            if step:
                research_plan.append(step)

        if not research_plan:
            raise PlannerError(
                "The planning model returned an empty research plan."
            )

        return {"research_plan": research_plan}
