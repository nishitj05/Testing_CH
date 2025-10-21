"""Wrapper around the Gemini 2.5 Flash API used to craft UI narratives."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore


@dataclass
class GeminiResponse:
    prompt: str
    content: str
    model: str = "gemini-2.5-flash"


class GeminiClient:
    """Thin client around the Gemini SDK with graceful fallbacks."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key and genai is not None)

    def generate(self, prompt: str) -> GeminiResponse:
        if not self.is_available():
            fallback = self._fallback(prompt)
            return GeminiResponse(prompt=prompt, content=fallback, model=self.model)

        response = genai.GenerativeModel(self.model).generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)
        return GeminiResponse(prompt=prompt, content=text, model=self.model)

    @staticmethod
    def _fallback(prompt: str) -> str:
        return (
            "[Gemini placeholder] "
            "Provide a concise business-friendly explanation summarising: "
            f"{prompt[:200]}..."
        )
