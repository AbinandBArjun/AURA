import re

from app.agents.state import ResearchState
from app.services.semantic_verifier import VerificationStatus


class ReportGeneratorError(Exception):
    """Raised when a research report cannot be generated."""


class ReportGeneratorNode:
    """Generates a deterministic, evidence-linked Markdown research report."""

    VALID_CLAIM_ID = re.compile(r"C[1-9]\d*")
    VALID_EVIDENCE_ID = re.compile(r"S[1-9]\d*")

    VALID_STATUSES = {
        status.value for status in VerificationStatus
    }

    @staticmethod
    def _required_text(value, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ReportGeneratorError(
                f"{field_name} must be a non-empty string."
            )
        return value.strip()

    @staticmethod
    def _escape_markdown(value: str) -> str:
        """Escape Markdown syntax in untrusted text."""
        return (
            value.replace("\\", "\\\\")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("`", "\\`")
            .replace("*", "\\*")
            .replace("_", "\\_")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("#", "\\#")
        )

    @classmethod
    def _format_untrusted_block(cls, value: str) -> str:
        """Render untrusted multiline text inside a Markdown blockquote."""
        escaped = cls._escape_markdown(value)

        return "\n".join(
            f"> {line}" if line else ">"
            for line in escaped.splitlines()
        )

    def __call__(self, state: ResearchState) -> dict:
        question = self._required_text(
            state.get("question"), "Research question"
        )
        findings = self._required_text(
            state.get("findings"), "Research findings"
        )

        claims = state.get("claims", [])
        verification_results = state.get("verification_results", [])
        evidence = state.get("evidence", [])

        if not isinstance(claims, list):
            raise ReportGeneratorError("Claims must be a list.")

        if not isinstance(verification_results, list):
            raise ReportGeneratorError(
                "Verification results must be a list."
            )

        if not isinstance(evidence, list):
            raise ReportGeneratorError("Evidence must be a list.")

        # Validate and index registered evidence.
        evidence_by_id = {}

        for item in evidence:
            if not isinstance(item, dict):
                raise ReportGeneratorError(
                    "Each evidence item must be a dictionary."
                )

            citation_id = item.get("citation_id")
            document_id = item.get("document_id")
            chunk_id = item.get("chunk_id")
            filename = item.get("filename")
            text = item.get("text")

            if (
                not isinstance(citation_id, str)
                or not self.VALID_EVIDENCE_ID.fullmatch(citation_id)
            ):
                raise ReportGeneratorError(
                    "Invalid registered evidence ID."
                )

            if citation_id in evidence_by_id:
                raise ReportGeneratorError(
                    f"Duplicate evidence ID: {citation_id}"
                )

            if (
                not isinstance(document_id, int)
                or isinstance(document_id, bool)
                or document_id <= 0
                or not isinstance(chunk_id, int)
                or isinstance(chunk_id, bool)
                or chunk_id <= 0
            ):
                raise ReportGeneratorError(
                    f"Invalid source identifiers for {citation_id}."
                )

            if not isinstance(filename, str) or not filename.strip():
                raise ReportGeneratorError(
                    f"Invalid filename for {citation_id}."
                )

            if not isinstance(text, str) or not text.strip():
                raise ReportGeneratorError(
                    f"Evidence text is empty for {citation_id}."
                )

            page_start = item.get("page_start")
            page_end = item.get("page_end")

            if page_start is not None and (
                not isinstance(page_start, int)
                or isinstance(page_start, bool)
                or page_start <= 0
            ):
                raise ReportGeneratorError(
                    f"Invalid starting page for {citation_id}."
                )

            if page_end is not None and (
                not isinstance(page_end, int)
                or isinstance(page_end, bool)
                or page_end <= 0
            ):
                raise ReportGeneratorError(
                    f"Invalid ending page for {citation_id}."
                )

            if (
                page_start is not None
                and page_end is not None
                and page_start > page_end
            ):
                raise ReportGeneratorError(
                    f"Invalid page range for {citation_id}."
                )

            evidence_by_id[citation_id] = item

        # Validate claims and their evidence references.
        claims_by_id = {}

        for claim in claims:
            if not isinstance(claim, dict):
                raise ReportGeneratorError(
                    "Each claim must be a dictionary."
                )

            claim_id = claim.get("claim_id")
            claim_text = claim.get("claim_text")
            evidence_ids = claim.get("evidence_ids")

            if (
                not isinstance(claim_id, str)
                or not self.VALID_CLAIM_ID.fullmatch(claim_id)
            ):
                raise ReportGeneratorError("Invalid claim ID.")

            if claim_id in claims_by_id:
                raise ReportGeneratorError(
                    f"Duplicate claim ID: {claim_id}"
                )

            if not isinstance(claim_text, str) or not claim_text.strip():
                raise ReportGeneratorError(
                    f"Claim text is empty: {claim_id}"
                )

            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ReportGeneratorError(
                    f"Claim {claim_id} must reference evidence."
                )

            if any(
                not isinstance(item, str)
                or not self.VALID_EVIDENCE_ID.fullmatch(item)
                for item in evidence_ids
            ):
                raise ReportGeneratorError(
                    f"Invalid evidence references for {claim_id}."
                )

            if len(evidence_ids) != len(set(evidence_ids)):
                raise ReportGeneratorError(
                    f"Duplicate evidence references for {claim_id}."
                )

            unknown_ids = set(evidence_ids) - set(evidence_by_id)

            if unknown_ids:
                raise ReportGeneratorError(
                    f"Claim {claim_id} references unknown evidence: "
                    f"{sorted(unknown_ids)}"
                )

            claims_by_id[claim_id] = {
                "claim_text": claim_text.strip(),
                "evidence_ids": evidence_ids,
            }

        # Validate one-to-one claim verification results.
        results_by_id = {}

        for result in verification_results:
            if not isinstance(result, dict):
                raise ReportGeneratorError(
                    "Each verification result must be a dictionary."
                )

            claim_id = result.get("claim_id")
            status = result.get("status")
            rationale = result.get("rationale")
            evidence_ids = result.get("evidence_ids")

            if not isinstance(claim_id, str) or claim_id not in claims_by_id:
                raise ReportGeneratorError(
                    f"Verification references unknown claim: {claim_id}"
                )

            if claim_id in results_by_id:
                raise ReportGeneratorError(
                    f"Duplicate verification result: {claim_id}"
                )

            if (
                not isinstance(status, str)
                or status not in self.VALID_STATUSES
            ):
                raise ReportGeneratorError(
                    f"Invalid verification status for {claim_id}."
                )

            if not isinstance(rationale, str) or not rationale.strip():
                raise ReportGeneratorError(
                    f"Verification rationale is empty: {claim_id}"
                )

            if not isinstance(evidence_ids, list):
                raise ReportGeneratorError(
                    f"Verification evidence IDs must be a list: {claim_id}"
                )

            if any(
                not isinstance(item, str)
                or not self.VALID_EVIDENCE_ID.fullmatch(item)
                for item in evidence_ids
            ):
                raise ReportGeneratorError(
                    f"Invalid verification evidence IDs: {claim_id}"
                )

            if len(evidence_ids) != len(set(evidence_ids)):
                raise ReportGeneratorError(
                    f"Duplicate verification evidence IDs: {claim_id}"
                )

            claim_evidence_ids = set(
                claims_by_id[claim_id]["evidence_ids"]
            )

            if not set(evidence_ids).issubset(claim_evidence_ids):
                raise ReportGeneratorError(
                    f"Verification references evidence not assigned "
                    f"to claim {claim_id}."
                )

            if (
                status != VerificationStatus.INSUFFICIENT_EVIDENCE.value
                and not evidence_ids
            ):
                raise ReportGeneratorError(
                    f"Verification evidence is required for {claim_id}."
                )

            results_by_id[claim_id] = {
                "status": status,
                "rationale": rationale.strip(),
                "evidence_ids": evidence_ids,
            }

        missing_results = set(claims_by_id) - set(results_by_id)

        if missing_results:
            raise ReportGeneratorError(
                "Missing verification results for claims: "
                f"{sorted(missing_results)}"
            )

        # Build the report without modifying LLM-generated findings.
        lines = [
            "# Research Report",
            "",
            "## Research Question",
            "",
            self._format_untrusted_block(question),
            "",
            "## Executive Summary",
            "",
            self._format_untrusted_block(findings),
            "",
            "## Claim Verification",
            "",
        ]

        if not claims_by_id:
            lines.append(
                "No verifiable claims were produced from the retrieved evidence."
            )

        status_labels = {
            "supported": "Supported",
            "partially_supported": "Partially supported",
            "contradicted": "Contradicted",
            "insufficient_evidence": "Insufficient evidence",
        }

        for claim_id, claim in claims_by_id.items():
            result = results_by_id[claim_id]

            lines.extend([
                f"### {claim_id}",
                "",
                self._format_untrusted_block(claim["claim_text"]),
                "",
                f"**Verification status:** "
                f"{status_labels[result['status']]}",
                "",
                "**Rationale:**",
                "",
                self._format_untrusted_block(result["rationale"]),
                "",
                "**Claim evidence:** "
                + ", ".join(
                    f"[{item}]" for item in claim["evidence_ids"]
                ),
                "",
                "**Verification evidence:** "
                + (
                    ", ".join(
                        f"[{item}]" for item in result["evidence_ids"]
                    )
                    if result["evidence_ids"]
                    else "No evidence identified."
                ),
                "",
            ])

        # Describe limitations based on actual verification outcomes.
        lines.extend([
            "## Limitations and Caveats",
            "",
        ])

        limited_claims = [
            (claim_id, results_by_id[claim_id])
            for claim_id in claims_by_id
            if results_by_id[claim_id]["status"] != "supported"
        ]

        if not claims_by_id:
            lines.append(
                "No claims could be verified because no claims were produced."
            )
        elif not limited_claims:
            lines.append(
                "All generated claims were classified as supported "
                "by the supplied evidence."
            )
        else:
            for claim_id, result in limited_claims:
                lines.extend([
                    f"- **{claim_id} - "
                    f"{status_labels[result['status']]}:**",
                    "",
                    self._format_untrusted_block(result["rationale"]),
                    "",
                ])

        # Include only evidence referenced by claims, preserving retrieval order.
        referenced_ids = {
            evidence_id
            for claim in claims_by_id.values()
            for evidence_id in claim["evidence_ids"]
        }

        lines.extend([
            "",
            "## Sources",
            "",
        ])

        sources = [
            (citation_id, item)
            for citation_id, item in evidence_by_id.items()
            if citation_id in referenced_ids
        ]

        if not sources:
            lines.append("No sources were linked to research claims.")
        else:
            for citation_id, item in sources:
                filename = self._escape_markdown(item["filename"])
                source_line = (
                    f"- **[{citation_id}]** {filename} "
                    f"(Document {item['document_id']}, "
                    f"Chunk {item['chunk_id']}"
                )

                page_start = item.get("page_start")
                page_end = item.get("page_end")

                if page_start is not None and page_end is not None:
                    if page_start == page_end:
                        source_line += f", Page {page_start}"
                    else:
                        source_line += f", Pages {page_start}-{page_end}"
                elif page_start is not None:
                    source_line += f", Page {page_start}"
                elif page_end is not None:
                    source_line += f", Ending page {page_end}"

                source_line += ")"
                lines.append(source_line)

        return {"report": "\n".join(lines).strip() + "\n"}
