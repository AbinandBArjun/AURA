import re


class SectionClassifier:
    """
    Tracks document sections and subsections across pages.
    Excludes references and endnotes from normal retrieval.
    """

    HEADING_PATTERNS = {
        "executive_summary": re.compile(
            r"^\s*executive\s+summary\s*$", re.IGNORECASE
        ),
        "abstract": re.compile(
            r"^\s*abstract\s*$", re.IGNORECASE
        ),
        "introduction": re.compile(
            r"^\s*(?:[IVX]+\.\s*)?introduction\s*$", re.IGNORECASE
        ),
        "results": re.compile(
            r"^\s*(?:[IVX]+\.\s*)?(?:results|findings)\s*$",
            re.IGNORECASE,
        ),
        "discussion": re.compile(
            r"^\s*(?:[IVX]+\.\s*)?discussion\s*$", re.IGNORECASE
        ),
        "conclusion": re.compile(
            r"^\s*(?:[IVX]+\.\s*)?conclusions?\s*$",
            re.IGNORECASE,
        ),
        "endnotes": re.compile(
            r"^\s*endnotes\s*$", re.IGNORECASE
        ),
        "references": re.compile(
            r"^\s*(?:references|bibliography)\s*$",
            re.IGNORECASE,
        ),
    }

    # Explicit headings identified in the uploaded research paper.
    CUSTOM_HEADING_PATTERNS = {
        "what_do_we_know": re.compile(
            r"^\s*II\.\s*What Do We Know\?\s*$", re.IGNORECASE
        ),
        "basic_statistics": re.compile(
            r"^\s*A\.\s*Basic Statistics\s*$", re.IGNORECASE
        ),
        "ai_productivity": re.compile(
            r"^\s*B\.\s*AI and Productivity\s*$", re.IGNORECASE
        ),
        "ai_labor": re.compile(
            r"^\s*C\.\s*AI and Labor\s*$", re.IGNORECASE
        ),
        "antitrust_data_portability": re.compile(
            r"^\s*III\.\s*AI,\s*Antitrust,\s*and Data Portability\s*$",
            re.IGNORECASE,
        ),
        "digital_economy_concentration": re.compile(
            r"^\s*A\.\s*Impact of the Digital Economy on Concentration Broadly\s*$",
            re.IGNORECASE,
        ),
        "large_datasets": re.compile(
            r"^\s*B\.\s*Large Data Sets as a Barrier to Entry\s*$",
            re.IGNORECASE,
        ),
        "ai_labor_markets": re.compile(
            r"^\s*IV\.\s*AI and Labor Markets.*$",
            re.IGNORECASE,
        ),
        "universal_basic_income": re.compile(
            r"^\s*A\.\s*Universal Basic Income\s*$", re.IGNORECASE
        ),
        "employment_subsidies": re.compile(
            r"^\s*B\.\s*Employment Subsidies\s*$", re.IGNORECASE
        ),
        "guaranteed_employment": re.compile(
            r"^\s*C\.\s*Guaranteed Employment\s*$", re.IGNORECASE
        ),
        "ai_specific_agency": re.compile(
            r"^\s*V\.\s*Is an AI[- ]Specific Agency Needed\?\s*$",
            re.IGNORECASE,
        ),
    }

    EXCLUDED_SECTIONS = {"references", "endnotes"}

    def detect_heading(self, line: str) -> str | None:
        normalized_line = line.strip()

        # Check document-specific headings first.
        for section, pattern in self.CUSTOM_HEADING_PATTERNS.items():
            if pattern.fullmatch(normalized_line):
                return section

        # Then check general headings.
        for section, pattern in self.HEADING_PATTERNS.items():
            if pattern.fullmatch(normalized_line):
                return section

        return None

    def classify(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"

        for line in text.splitlines():
            heading = self.detect_heading(line)
            if heading:
                return heading

        return "body"

    def is_retrievable(self, section: str) -> bool:
        return section not in self.EXCLUDED_SECTIONS

    def annotate_pages(self, pages: list[dict]) -> list[dict]:
        """
        Annotate text segments with their current section.

        Section state continues across page boundaries.
        """

        current_section = "body"
        segments = []

        for page in pages:
            page_number = page["page_number"]

            for line in page["text"].splitlines():
                heading = self.detect_heading(line)

                if heading:
                    current_section = heading

                if not line.strip():
                    continue

                segments.append({
                    "page_number": page_number,
                    "text": line,
                    "section": current_section,
                    "retrievable": self.is_retrievable(current_section),
                })

        return segments
