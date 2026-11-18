from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.research_run import ResearchRun
from app.models.workspace import Workspace
from app.schemas.research import ResearchResponse


class ResearchExecutionError(Exception):
    """Raised when the research workflow fails."""


class ResearchService:
    """Coordinates research execution and persists its lifecycle."""

    def __init__(self, db: Session, graph):
        self.db = db
        self.graph = graph

    def run(
        self,
        workspace_id: int,
        question: str,
    ) -> ResearchResponse:
        if not isinstance(workspace_id, int) or isinstance(workspace_id, bool):
            raise ValueError("Invalid workspace ID.")

        if workspace_id <= 0:
            raise ValueError("Invalid workspace ID.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question cannot be empty.")

        workspace = self.db.get(Workspace, workspace_id)

        if workspace is None:
            raise ValueError("Workspace not found.")

        research_run = ResearchRun(
            workspace_id=workspace_id,
            query=question,
            status="running",
            graph_version="v0.1",
            model_config_json={},
            started_at=datetime.now(timezone.utc),
        )

        self.db.add(research_run)
        self.db.commit()
        self.db.refresh(research_run)

        try:
            state = self.graph.invoke({
                "workspace_id": workspace_id,
                "question": question,
            })

            if not isinstance(state, dict):
                raise ValueError("Research workflow returned invalid state.")

            evidence = state.get("evidence", [])
            claims = state.get("claims", [])
            verification_results = state.get(
                "verification_results", []
            )

            if not isinstance(evidence, list):
                raise ValueError("Invalid research evidence.")

            if not isinstance(claims, list):
                raise ValueError("Invalid research claims.")

            if not isinstance(verification_results, list):
                raise ValueError("Invalid verification results.")

            referenced_ids = {
                evidence_id
                for claim in claims
                for evidence_id in claim["evidence_ids"]
            }

            sources = [
                {
                    "citation_id": item["citation_id"],
                    "document_id": item["document_id"],
                    "filename": item["filename"],
                    "chunk_id": item["chunk_id"],
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": item["score"],
                }
                for item in evidence
                if item["citation_id"] in referenced_ids
            ]

            findings = state.get("findings", "")
            report = state.get("report", "")

            response = ResearchResponse.model_validate({
                "research_run_id": research_run.id,
                "status": "completed",
                "question": question,
                "research_plan": state.get("research_plan", []),
                "findings": findings,
                "report": report,
                "claims": claims,
                "verification_results": verification_results,
                "sources": sources,
                "evidence_count": len(evidence),
            })

            research_run.status = "completed"
            research_run.result_summary = findings[:1000]
            research_run.completed_at = datetime.now(timezone.utc)

            self.db.commit()

            return response

        except Exception as exc:
            self.db.rollback()

            failed_run = self.db.get(
                ResearchRun,
                research_run.id,
            )

            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.error_message = (
                    "Research workflow execution failed."
                )
                failed_run.completed_at = datetime.now(timezone.utc)

                self.db.commit()

            raise ResearchExecutionError(
                "Research workflow execution failed."
            ) from exc
