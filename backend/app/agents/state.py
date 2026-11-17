from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Shared state passed between AURA research workflow nodes."""

    # Request context
    workspace_id: int
    question: str

    # Planning
    research_plan: list[str]

    # Retrieved evidence
    evidence: list[dict[str, Any]]

    # Analysis and verification
    findings: str
    claims: list[dict[str, Any]]
    verification_results: list[dict[str, Any]]

    # Final output
    report: str

    # Workflow diagnostics
    errors: list[str]
