from app.agents.state import ResearchState
from app.services.search_service import SearchService


class ResearcherError(Exception):
    """Raised when evidence retrieval fails."""


class ResearcherNode:
    """Retrieves evidence for each step in the research plan."""

    def __init__(
        self,
        search_service: SearchService | None = None,
        result_limit: int = 5,
    ):
        if result_limit <= 0:
            raise ValueError("Result limit must be positive.")

        self.search_service = (
            search_service
            if search_service is not None
            else SearchService()
        )
        self.result_limit = result_limit

    def __call__(self, state: ResearchState) -> dict:
        workspace_id = state.get("workspace_id")
        research_plan = state.get("research_plan", [])

        if (
            not isinstance(workspace_id, int)
            or isinstance(workspace_id, bool)
            or workspace_id <= 0
        ):
            raise ResearcherError("A valid workspace ID is required.")

        if not isinstance(research_plan, list) or not any(
            isinstance(step, str) and step.strip()
            for step in research_plan
        ):
            raise ResearcherError("Research plan cannot be empty.")

        evidence = []
        seen_chunks = set()

        for step in research_plan:
            if not isinstance(step, str) or not step.strip():
                continue

            query = step.strip()

            try:
                results = self.search_service.search(
                    workspace_id=workspace_id,
                    query=query,
                    limit=self.result_limit,
                    mode="hybrid_rerank",
                )
            except Exception as exc:
                raise ResearcherError(
                    f"Evidence retrieval failed for research step: {query}"
                ) from exc

            if not isinstance(results, list):
                raise ResearcherError(
                    f"Invalid retrieval response for research step: {query}"
                )

            for result in results:
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("document_id"), int)
                    or isinstance(result.get("document_id"), bool)
                    or result.get("document_id", 0) <= 0
                    or not isinstance(result.get("chunk_id"), int)
                    or isinstance(result.get("chunk_id"), bool)
                    or result.get("chunk_id", 0) <= 0
                ):
                    raise ResearcherError(
                        f"Invalid evidence metadata for research step: {query}"
                    )

                key = (result["document_id"], result["chunk_id"])

                try:
                    if key in seen_chunks:
                        continue
                    seen_chunks.add(key)
                except TypeError as exc:
                    raise ResearcherError(
                        "Evidence document and chunk IDs must be hashable."
                    ) from exc

                evidence.append({
                    **result,
                    "research_query": query,
                })

        return {"evidence": evidence}
