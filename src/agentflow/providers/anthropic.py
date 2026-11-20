"""Anthropic Messages API adapter (Claude)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..errors import ConfigurationError, ProviderError
from ..types import Message, ProviderResponse, ToolCall, ToolSpec, Usage
from .base import HTTPProvider

ANTHROPIC_VERSION = "2023-06-01"


def to_anthropic_payload(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split out the system prompt and convert messages to Anthropic content blocks.

    Tool results become ``tool_result`` blocks inside a user turn; consecutive results are
    merged into a single user message as the API requires.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in m.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
                "is_error": m.is_error,
            }
            last = out[-1] if out else None
            if last and last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": [{"type": "text", "text": m.content}]})
    return "\n\n".join(system_parts), out


def from_anthropic_response(data: dict[str, Any]) -> ProviderResponse:
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ProviderError("anthropic: response has no content list")
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    calls = [
        ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    usage = data.get("usage") or {}
    return ProviderResponse(
        message=Message.assistant(text, calls),
        stop_reason=data.get("stop_reason"),
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ),
        model=data.get("model"),
    )


class AnthropicProvider(HTTPProvider):
    """Calls ``POST {base_url}/v1/messages``."""

    name = "anthropic"
    default_model = "claude-3-5-haiku-20241022"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            client=client,
            **kwargs,
        )
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs: Any) -> AnthropicProvider:
        api_key = os.environ.get(cls.api_key_env)
        if not api_key:
            raise ConfigurationError(f"anthropic: set {cls.api_key_env} to use this provider")
        return cls(api_key=api_key, model=model or os.environ.get("AGENTFLOW_MODEL"), **kwargs)

    def build_payload(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        system, converted = to_anthropic_payload(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        data = self._post("/v1/messages", self.build_payload(messages, tools))
        return from_anthropic_response(data)
