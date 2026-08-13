"""
generation/llm_client.py
========================
Thin LLM Invocation Layer.
Provides a clean abstraction boundary between domain logic and LLM APIs,
supporting both real runtime providers (Google Gemini) and a deterministic FakeLLMClient for testing.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, Sequence, Any


class LLMClient(Protocol):
    """Protocol definition for thin LLM invocation layer."""

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        ...


class FakeLLMClient:
    """
    Deterministic stub LLM client for testing and offline execution.
    Does NOT make live network API calls.
    """

    def __init__(
        self,
        canned_response: str | None = None,
        field_responses: dict[str, str] | None = None,
        generator_fn: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self.canned_response = canned_response
        self.field_responses = field_responses or {}
        self.generator_fn = generator_fn
        self.calls: list[dict[str, str | None]] = []

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})

        if self.generator_fn:
            return self.generator_fn(prompt, system_instruction)

        # Check if prompt targets a specific field_id matching field_responses
        for fid, resp in self.field_responses.items():
            if f"TARGET SECTION: [{fid}]" in prompt or f"Section {fid}" in prompt:
                return resp

        if self.canned_response is not None:
            return self.canned_response

        # Default fallback response echoing confirmed information and standard citation
        return (
            "The system shall support the specified project requirements. "
            "Grounded by reference standards [R1]."
        )


class GeminiLLMClient:
    """
    Real LLM client provider using Google Generative AI (Gemini API).
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash") -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. Please install it using 'pip install google-generativeai'."
            )

        genai.configure(api_key=key)
        self.model_name = model_name

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )
        response = model.generate_content(prompt)
        return response.text or ""


def get_default_llm_client(fallback_fake: bool = True) -> LLMClient:
    """
    Factory function to retrieve the configured LLM client.
    Returns GeminiLLMClient if API key is present in env, otherwise returns FakeLLMClient if fallback_fake is True.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            return GeminiLLMClient(api_key=api_key)
        except Exception:
            if not fallback_fake:
                raise
    if fallback_fake:
        return FakeLLMClient()
    raise ValueError("No LLM provider configured and fallback_fake is False.")
