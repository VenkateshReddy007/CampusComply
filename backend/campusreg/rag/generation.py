"""Text generation protocol and Google GenAI Gemma adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol

from campusreg.config import RAGSettings
from campusreg.rag.errors import GenerationError, MissingAPIConfigurationError


class TextGenerator(Protocol):
    """Generation interface used by answer(), replaceable in unit tests."""

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system_instruction: str,
        temperature: float,
    ) -> str:
        """Generate text from the provided context-grounded prompt."""


class GoogleGenAIProvider:
    """Google GenAI SDK adapter; credentials are read from the process env."""

    def __init__(self, client: object) -> None:
        self._client = client

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GoogleGenAIProvider":
        """Create a provider without printing or retaining the API-key string."""

        values = os.environ if environ is None else environ
        api_key = (
            values.get("GOOGLE_API_KEY")
            or values.get("GEMMA_API_KEY")
            or values.get("gemma_API_Key")
        )
        if not api_key or not api_key.strip():
            raise MissingAPIConfigurationError(
                "Set GOOGLE_API_KEY or GEMMA_API_KEY before using Gemma generation."
            )
        try:
            from google import genai

            client = genai.Client(api_key=api_key.strip())
        except ImportError as exc:
            raise RuntimeError(
                "Install campusreg-rag with the [google] extra to use Gemma generation."
            ) from exc
        except Exception:
            raise RuntimeError("The Google GenAI client could not be initialized.") from None
        return cls(client)

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system_instruction: str,
        temperature: float,
    ) -> str:
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
        except Exception:
            raise GenerationError("Gemma generation request failed.") from None

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("Gemma returned no answer text.", code="empty_generation")
        return text.strip()


def generate(
    user_text: str,
    generator: TextGenerator,
    settings: RAGSettings | None = None,
    *,
    system_text: str,
) -> str:
    """Call an injected provider and normalize all failures to GenerationError."""

    selected = settings or RAGSettings.baseline()
    try:
        response = generator.generate(
            user_text,
            model=selected.gemma_model,
            system_instruction=system_text,
            temperature=selected.temperature,
        )
    except GenerationError:
        raise
    except Exception:
        raise GenerationError("Gemma generation request failed.") from None
    if not isinstance(response, str) or not response.strip():
        raise GenerationError("Gemma returned no answer text.", code="empty_generation")
    return response.strip()
