from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerNode
from app.agents.researcher import ResearcherNode
from app.agents.state import ResearchState
from app.services.search_service import SearchService


def build_research_graph(
    planner: PlannerNode | None = None,
    researcher: ResearcherNode | None = None,
):
    """Build the AURA research workflow."""

    planner_node = planner if planner is not None else PlannerNode()

    researcher_node = (
        researcher
        if researcher is not None
        else ResearcherNode(search_service=SearchService())
    )

    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", END)

    return builder.compile()
