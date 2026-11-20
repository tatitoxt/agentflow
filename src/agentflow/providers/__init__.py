"""LLM providers behind a single :class:`Provider` interface."""

from __future__ import annotations

import os

from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .base import HTTPProvider, Provider
from .fake import FakeProvider
from .openai import GroqProvider, OpenAIProvider

PROVIDER_NAMES = ("fake", "openai", "anthropic", "groq")


def provider_from_env(name: str | None = None, *, fake: Provider | None = None) -> Provider:
    """Build a provider from ``AGENTFLOW_PROVIDER`` (default ``fake``).

    Real providers are opt-in: they need ``AGENTFLOW_PROVIDER`` (or ``name``) set *and* their
    API key env var. ``fake`` returns the supplied scripted provider.
    """
    selected = (name or os.environ.get("AGENTFLOW_PROVIDER") or "fake").lower()
    if selected == "fake":
        if fake is None:
            raise ConfigurationError("fake provider selected but no script was supplied")
        return fake
    if selected == "openai":
        return OpenAIProvider.from_env()
    if selected == "groq":
        return GroqProvider.from_env()
    if selected == "anthropic":
        return AnthropicProvider.from_env()
    raise ConfigurationError(f"unknown provider {selected!r}; choose one of {PROVIDER_NAMES}")


__all__ = [
    "PROVIDER_NAMES",
    "AnthropicProvider",
    "FakeProvider",
    "GroqProvider",
    "HTTPProvider",
    "OpenAIProvider",
    "Provider",
    "provider_from_env",
]
