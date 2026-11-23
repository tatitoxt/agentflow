"""Opt-in smoke tests against real provider APIs.

Skipped unless the matching API key is set; meant for the manual
`live-smoke` workflow, never for CI gating.
"""

import asyncio
import os

import pytest

from agentflow import Agent, Message
from agentflow.providers.anthropic import AnthropicProvider
from agentflow.providers.openai import OpenAIProvider

pytestmark = pytest.mark.live

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))

PROMPT = "Reply with exactly the word 'pong' and nothing else."


@pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
class TestOpenAILive:
    def test_complete_returns_text_and_usage(self):
        provider = OpenAIProvider.from_env()
        resp = provider.complete([Message.user(PROMPT)], tools=[])
        assert resp.text.strip(), "empty completion"
        assert resp.usage.input_tokens > 0 and resp.usage.output_tokens > 0

    def test_stream_yields_deltas(self):
        provider = OpenAIProvider.from_env()
        deltas: list[str] = []
        resp = provider.stream([Message.user(PROMPT)], tools=[], on_delta=deltas.append)
        assert deltas, "no stream deltas"
        assert resp.text.strip()
        assert resp.usage.output_tokens > 0

    def test_agent_arun_against_real_api(self):
        provider = OpenAIProvider.from_env()
        agent = Agent(provider=provider, system="You are terse.")
        result = asyncio.run(agent.arun(PROMPT))
        assert result.text.strip()
        assert result.usage.output_tokens > 0


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicLive:
    def test_complete_returns_text_and_usage(self):
        provider = AnthropicProvider.from_env()
        resp = provider.complete([Message.user(PROMPT)], tools=[])
        assert resp.text.strip(), "empty completion"
        assert resp.usage.input_tokens > 0 and resp.usage.output_tokens > 0

    def test_stream_yields_deltas(self):
        provider = AnthropicProvider.from_env()
        deltas: list[str] = []
        resp = provider.stream([Message.user(PROMPT)], tools=[], on_delta=deltas.append)
        assert deltas, "no stream deltas"
        assert resp.text.strip()

    def test_agent_arun_against_real_api(self):
        provider = AnthropicProvider.from_env()
        agent = Agent(provider=provider, system="You are terse.")
        result = asyncio.run(agent.arun(PROMPT))
        assert result.text.strip()
