"""LLM providers behind a single :class:`Provider` interface."""

from __future__ import annotations

import os

from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .base import HTTPProvider, Provider
from .fake import FakeProvider
from .openai import GroqProvider, OllamaProvider, OpenAIProvider
from .replay import RecordingProvider, ReplayProvider

PROVIDER_NAMES = ("fake", "openai", "anthropic", "groq", "ollama", "replay")


def provider_from_env(name: str | None = None, *, fake: Provider | None = None) -> Provider:
    """Build a provider from ``AGENTFLOW_PROVIDER`` (default ``fake``).

    Real providers are opt-in: they need ``AGENTFLOW_PROVIDER`` (or ``name``) set *and* their
    API key env var. ``fake`` returns the supplied scripted provider.
    """
    selected = (name or os.environ.get("AGENTFLOW_PROVIDER") or "fake").lower()
    provider: Provider
    if selected == "fake":
        if fake is None:
            raise ConfigurationError("fake provider selected but no script was supplied")
        provider = fake
    elif selected == "openai":
        provider = OpenAIProvider.from_env()
    elif selected == "groq":
        provider = GroqProvider.from_env()
    elif selected == "ollama":
        provider = OllamaProvider.from_env()
    elif selected == "anthropic":
        provider = AnthropicProvider.from_env()
    elif selected == "replay":
        provider = ReplayProvider(
            os.environ.get("AGENTFLOW_REPLAY_FILE") or "agentflow-replay.jsonl"
        )
    else:
        raise ConfigurationError(f"unknown provider {selected!r}; choose one of {PROVIDER_NAMES}")
    if record_file := os.environ.get("AGENTFLOW_RECORD_FILE"):
        provider = RecordingProvider(provider, record_file)
    return provider


__all__ = [
    "PROVIDER_NAMES",
    "AnthropicProvider",
    "FakeProvider",
    "GroqProvider",
    "OllamaProvider",
    "RecordingProvider",
    "ReplayProvider",
    "HTTPProvider",
    "OpenAIProvider",
    "Provider",
    "provider_from_env",
]
