from app.agents.state import ResearchState


class EvidenceRegistryError(Exception):
    """Raised when retrieved evidence cannot be registered."""


class EvidenceRegistryNode:
    """Assigns server-generated citation IDs to retrieved evidence."""

    def __call__(self, state: ResearchState) -> dict:
        evidence = state.get("evidence", [])

        if not isinstance(evidence, list):
            raise EvidenceRegistryError(
                "Evidence must be a list."
            )

        registered_evidence = []

        for index, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                raise EvidenceRegistryError(
                    "Each evidence item must be a dictionary."
                )

            if (
                not isinstance(item.get("text"), str)
                or not item["text"].strip()
            ):
                raise EvidenceRegistryError(
                    "Evidence text cannot be empty."
                )

            registered_evidence.append({
                **item,
                "citation_id": f"S{index}",
            })

        return {"evidence": registered_evidence}
