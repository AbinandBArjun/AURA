import json

import pytest

from app.agents.analyst import AnalystError, AnalystNode


class FakeOllamaClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def evidence():
    return [
        {
            "citation_id": "S1",
            "document_id": 1,
            "filename": "research.pdf",
            "chunk_id": 10,
            "text": "Hybrid retrieval combines dense and sparse search.",
            "page_start": 1,
            "page_end": 2,
            "score": 0.85,
        },
        {
            "citation_id": "S2",
            "document_id": 1,
            "filename": "research.pdf",
            "chunk_id": 11,
            "text": "Dense retrieval uses semantic embeddings.",
            "page_start": 2,
            "page_end": 3,
            "score": 0.80,
        },
    ]


@pytest.fixture
def valid_response():
    return {
        "findings": "Hybrid retrieval combines dense and sparse search.",
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": (
                    "Hybrid retrieval combines dense and sparse search."
                ),
                "evidence_ids": ["S1"],
            },
            {
                "claim_id": "C2",
                "claim_text": (
                    "Dense retrieval uses semantic embeddings."
                ),
                "evidence_ids": ["S2"],
            },
        ],
    }


def test_analyst_generates_findings_and_claims(evidence, valid_response):
    client = FakeOllamaClient(json.dumps(valid_response))
    analyst = AnalystNode(client)

    result = analyst({
        "question": "Explain hybrid retrieval.",
        "evidence": evidence,
    })

    assert result["findings"] == valid_response["findings"]
    assert len(result["claims"]) == 2
    assert result["claims"][0]["evidence_ids"] == ["S1"]
    assert result["claims"][1]["evidence_ids"] == ["S2"]


def test_analyst_preserves_evidence_ids_in_prompt(evidence, valid_response):
    client = FakeOllamaClient(json.dumps(valid_response))
    analyst = AnalystNode(client)

    analyst({
        "question": "Explain hybrid retrieval.",
        "evidence": evidence,
    })

    prompt = client.calls[0]["prompt"]

    assert '"citation_id": "S1"' in prompt
    assert '"citation_id": "S2"' in prompt


def test_analyst_uses_low_temperature(evidence, valid_response):
    client = FakeOllamaClient(json.dumps(valid_response))
    analyst = AnalystNode(client)

    analyst({
        "question": "Explain hybrid retrieval.",
        "evidence": evidence,
    })

    assert client.calls[0]["temperature"] == 0.2


def test_analyst_rejects_empty_question(evidence):
    analyst = AnalystNode(FakeOllamaClient())

    with pytest.raises(AnalystError, match="question"):
        analyst({
            "question": " ",
            "evidence": evidence,
        })


def test_analyst_returns_insufficient_evidence_without_llm_call():
    client = FakeOllamaClient()
    analyst = AnalystNode(client)

    result = analyst({
        "question": "Explain hybrid retrieval.",
        "evidence": [],
    })

    assert result["claims"] == []
    assert "Insufficient evidence" in result["findings"]
    assert client.calls == []


def test_analyst_rejects_non_list_evidence():
    analyst = AnalystNode(FakeOllamaClient())

    with pytest.raises(AnalystError, match="Evidence must be a list"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": "invalid",
        })


@pytest.mark.parametrize("invalid_item", [None, "invalid", 123])
def test_analyst_rejects_invalid_evidence_items(invalid_item):
    analyst = AnalystNode(FakeOllamaClient())

    with pytest.raises(AnalystError, match="dictionary"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": [invalid_item],
        })


@pytest.mark.parametrize("citation_id", ["S0", "S-1", "invalid", None])
def test_analyst_rejects_invalid_registered_evidence_ids(citation_id):
    analyst = AnalystNode(FakeOllamaClient())

    item = {
        "citation_id": citation_id,
        "text": "Some evidence.",
    }

    with pytest.raises(AnalystError, match="registered evidence ID"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": [item],
        })


def test_analyst_rejects_duplicate_registered_evidence_ids():
    analyst = AnalystNode(FakeOllamaClient())

    item = {
        "citation_id": "S1",
        "text": "Some evidence.",
    }

    with pytest.raises(AnalystError, match="Duplicate"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": [item, item],
        })


def test_analyst_rejects_empty_evidence_text():
    analyst = AnalystNode(FakeOllamaClient())

    with pytest.raises(AnalystError, match="Evidence text"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": [{
                "citation_id": "S1",
                "text": " ",
            }],
        })


def test_analyst_rejects_malformed_json(evidence):
    analyst = AnalystNode(FakeOllamaClient("not valid JSON"))

    with pytest.raises(AnalystError, match="invalid structured response"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_rejects_unknown_evidence_id(evidence, valid_response):
    valid_response["claims"][0]["evidence_ids"] = ["S999"]

    analyst = AnalystNode(FakeOllamaClient(json.dumps(valid_response)))

    with pytest.raises(AnalystError, match="unknown evidence ID"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_rejects_duplicate_claim_ids(evidence, valid_response):
    valid_response["claims"][1]["claim_id"] = "C1"

    analyst = AnalystNode(FakeOllamaClient(json.dumps(valid_response)))

    with pytest.raises(AnalystError, match="Duplicate claim IDs"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_rejects_claim_without_evidence_ids(evidence, valid_response):
    valid_response["claims"][0]["evidence_ids"] = []

    analyst = AnalystNode(FakeOllamaClient(json.dumps(valid_response)))

    with pytest.raises(AnalystError, match="structured response"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_rejects_extra_response_fields(evidence, valid_response):
    valid_response["unexpected"] = "not allowed"

    analyst = AnalystNode(FakeOllamaClient(json.dumps(valid_response)))

    with pytest.raises(AnalystError, match="structured response"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_wraps_llm_errors(evidence):
    client = FakeOllamaClient(error=RuntimeError("Connection failed"))
    analyst = AnalystNode(client)

    with pytest.raises(AnalystError, match="Failed to generate"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })


def test_analyst_rejects_non_string_llm_response(evidence):
    analyst = AnalystNode(FakeOllamaClient(response=None))

    with pytest.raises(AnalystError, match="invalid response"):
        analyst({
            "question": "Explain hybrid retrieval.",
            "evidence": evidence,
        })
