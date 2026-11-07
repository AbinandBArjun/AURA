from unittest.mock import patch

import httpx
import pytest

from app.llm.ollama_client import OllamaClient, OllamaError


def test_generate_success():
    client = OllamaClient()

    mock_response = httpx.Response(
        200,
        json={"response": "RAG combines retrieval and generation."},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )

    with patch("app.llm.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        result = client.generate(
            prompt="Explain RAG.",
            system_prompt="You are a research assistant.",
        )

    assert result == "RAG combines retrieval and generation."


def test_empty_prompt_is_rejected():
    client = OllamaClient()

    with pytest.raises(ValueError):
        client.generate("   ")


def test_invalid_temperature_is_rejected():
    client = OllamaClient()

    with pytest.raises(ValueError):
        client.generate("Explain RAG.", temperature=3)


def test_connection_failure():
    client = OllamaClient()

    with patch("app.llm.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = (
            httpx.ConnectError("Connection refused")
        )

        with pytest.raises(OllamaError, match="Could not connect"):
            client.generate("Explain RAG.")
