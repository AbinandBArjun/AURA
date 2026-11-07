import httpx

from app.core.config import settings


class OllamaError(Exception):
    """Raised when Ollama requests fail."""


class OllamaClient:

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (
            base_url or settings.ollama_base_url
        ).rstrip("/")

        self.model = model or settings.ollama_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )

                response.raise_for_status()

            data = response.json()
            generated_text = data.get("response", "")

            if not generated_text.strip():
                raise OllamaError(
                    "Ollama returned an empty response."
                )

            return generated_text.strip()

        except httpx.TimeoutException as exc:
            raise OllamaError(
                "Ollama request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise OllamaError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        except ValueError as exc:
            raise OllamaError(
                "Ollama returned an invalid response."
            ) from exc
