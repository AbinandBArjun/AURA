from app.chunking.section_classifier import SectionClassifier


def test_detects_executive_summary():
    classifier = SectionClassifier()
    assert classifier.classify("Executive Summary\nAI is changing the economy.") == "executive_summary"


def test_detects_introduction():
    classifier = SectionClassifier()
    assert classifier.classify("I. Introduction\nArtificial intelligence is advancing.") == "introduction"


def test_detects_conclusion():
    classifier = SectionClassifier()
    assert classifier.classify("VI. Conclusion\nAI has productivity potential.") == "conclusion"


def test_detects_references():
    classifier = SectionClassifier()
    assert classifier.classify("References\nAuthor, 2019. Research title.") == "references"


def test_detects_endnotes():
    classifier = SectionClassifier()
    assert classifier.classify("Endnotes\n1. Supporting information.") == "endnotes"


def test_unknown_empty_text():
    classifier = SectionClassifier()
    assert classifier.classify("") == "unknown"


def test_references_are_not_retrievable():
    classifier = SectionClassifier()
    assert classifier.is_retrievable("references") is False
    assert classifier.is_retrievable("endnotes") is False
    assert classifier.is_retrievable("conclusion") is True


def test_section_state_continues_across_pages():
    classifier = SectionClassifier()

    pages = [
        {
            "page_number": 1,
            "text": "Executive Summary\nAI affects productivity.",
        },
        {
            "page_number": 2,
            "text": "AI also affects employment.",
        },
    ]

    segments = classifier.annotate_pages(pages)

    assert segments[0]["section"] == "executive_summary"
    assert segments[-1]["section"] == "executive_summary"


def test_references_remain_excluded_across_pages():
    classifier = SectionClassifier()

    pages = [
        {
            "page_number": 1,
            "text": "References\nAuthor One, 2019.",
        },
        {
            "page_number": 2,
            "text": "Author Two, 2020.",
        },
    ]

    segments = classifier.annotate_pages(pages)

    assert all(
        segment["retrievable"] is False
        for segment in segments
    )


def test_conclusion_and_endnotes_are_separated():
    classifier = SectionClassifier()

    pages = [
        {
            "page_number": 1,
            "text": (
                "VI. Conclusion\n"
                "AI may improve productivity.\n"
                "Endnotes\n"
                "1. Supporting information."
            ),
        }
    ]

    segments = classifier.annotate_pages(pages)

    conclusion = next(
        segment for segment in segments
        if "AI may improve" in segment["text"]
    )
    endnote = next(
        segment for segment in segments
        if "Supporting information" in segment["text"]
    )

    assert conclusion["section"] == "conclusion"
    assert conclusion["retrievable"] is True
    assert endnote["section"] == "endnotes"
    assert endnote["retrievable"] is False


def test_detects_new_major_sections():
    classifier = SectionClassifier()

    assert classifier.detect_heading("II. What Do We Know?") == "what_do_we_know"
    assert classifier.detect_heading(
        "III. AI, Antitrust, and Data Portability"
    ) == "antitrust_data_portability"
    assert classifier.detect_heading(
        "IV. AI and Labor Markets: UBI, Wage Supplements, and Guaranteed Employment"
    ) == "ai_labor_markets"
    assert classifier.detect_heading(
        "V. Is an AI-Specific Agency Needed?"
    ) == "ai_specific_agency"


def test_detects_subsections():
    classifier = SectionClassifier()

    assert classifier.detect_heading("A. Basic Statistics") == "basic_statistics"
    assert classifier.detect_heading("B. AI and Productivity") == "ai_productivity"
    assert classifier.detect_heading("C. AI and Labor") == "ai_labor"
    assert classifier.detect_heading("A. Universal Basic Income") == "universal_basic_income"
    assert classifier.detect_heading("B. Employment Subsidies") == "employment_subsidies"
    assert classifier.detect_heading("C. Guaranteed Employment") == "guaranteed_employment"


def test_custom_sections_remain_retrievable():
    classifier = SectionClassifier()

    assert classifier.is_retrievable("ai_productivity") is True
    assert classifier.is_retrievable("guaranteed_employment") is True
    assert classifier.is_retrievable("references") is False
    assert classifier.is_retrievable("endnotes") is False


def test_section_transitions_across_pages():
    classifier = SectionClassifier()

    pages = [
        {
            "page_number": 1,
            "text": (
                "II. What Do We Know?\n"
                "General discussion.\n"
                "B. AI and Productivity\n"
                "Productivity evidence."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "More productivity evidence.\n"
                "C. AI and Labor\n"
                "Labor evidence."
            ),
        },
    ]

    segments = classifier.annotate_pages(pages)

    assert segments[0]["section"] == "what_do_we_know"
    assert segments[1]["section"] == "what_do_we_know"
    assert segments[2]["section"] == "ai_productivity"
    assert segments[4]["section"] == "ai_productivity"
    assert segments[5]["section"] == "ai_labor"
    assert segments[6]["section"] == "ai_labor"
