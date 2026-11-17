from langgraph.graph import END, START, StateGraph

from app.agents.analyst import AnalystNode
from app.agents.evidence_registry import EvidenceRegistryNode
from app.agents.fact_checker import FactCheckerNode
from app.agents.planner import PlannerNode
from app.agents.report_generator import ReportGeneratorNode
from app.agents.researcher import ResearcherNode
from app.agents.state import ResearchState
from app.services.search_service import SearchService


def build_research_graph(
    planner: PlannerNode | None = None,
    researcher: ResearcherNode | None = None,
    evidence_registry: EvidenceRegistryNode | None = None,
    analyst: AnalystNode | None = None,
    fact_checker: FactCheckerNode | None = None,
    report_generator: ReportGeneratorNode | None = None,
):
    """Build the complete evidence-grounded AURA research workflow."""

    planner_node = planner if planner is not None else PlannerNode()

    researcher_node = (
        researcher
        if researcher is not None
        else ResearcherNode(search_service=SearchService())
    )

    evidence_registry_node = (
        evidence_registry
        if evidence_registry is not None
        else EvidenceRegistryNode()
    )

    analyst_node = analyst if analyst is not None else AnalystNode()

    fact_checker_node = (
        fact_checker
        if fact_checker is not None
        else FactCheckerNode()
    )

    report_generator_node = (
        report_generator
        if report_generator is not None
        else ReportGeneratorNode()
    )

    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("evidence_registry", evidence_registry_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("report_generator", report_generator_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "evidence_registry")
    builder.add_edge("evidence_registry", "analyst")
    builder.add_edge("analyst", "fact_checker")
    builder.add_edge("fact_checker", "report_generator")
    builder.add_edge("report_generator", END)

    return builder.compile()
