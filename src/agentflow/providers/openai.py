"""OpenAI Chat Completions adapter (also used for OpenAI-compatible APIs such as Groq)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx

from ..errors import ConfigurationError, ProviderError
from ..types import Message, ProviderResponse, ToolCall, ToolSpec, Usage
from .base import HTTPProvider


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            out.append(item)
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"_raw_arguments": raw}
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": raw}


def from_openai_response(data: dict[str, Any]) -> ProviderResponse:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("openai: response has no choices[0].message") from exc
    calls = [
        ToolCall(
            id=c["id"],
            name=c["function"]["name"],
            arguments=_parse_arguments(c["function"].get("arguments")),
        )
        for c in msg.get("tool_calls") or []
    ]
    usage = data.get("usage") or {}
    return ProviderResponse(
        message=Message.assistant(msg.get("content") or "", calls),
        stop_reason=choice.get("finish_reason"),
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
        model=data.get("model"),
    )


class OpenAIProvider(HTTPProvider):
    """Calls ``POST {base_url}/chat/completions``."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url or self.default_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            client=client,
            **kwargs,
        )
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs: Any) -> OpenAIProvider:
        api_key = os.environ.get(cls.api_key_env)
        if not api_key:
            raise ConfigurationError(f"{cls.name}: set {cls.api_key_env} to use this provider")
        return cls(api_key=api_key, model=model or os.environ.get("AGENTFLOW_MODEL"), **kwargs)

    def build_payload(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": to_openai_messages(messages)}
        if tools:
            payload["tools"] = to_openai_tools(tools)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        data = self._post("/chat/completions", self.build_payload(messages, tools))
        return from_openai_response(data)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        payload = self.build_payload(messages, tools)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        text_parts: list[str] = []
        tool_fragments: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = Usage()
        model: str | None = None
        for chunk in self._stream_events("/chat/completions", payload):
            model = model or chunk.get("model")
            raw_usage = chunk.get("usage")
            if raw_usage:
                usage = Usage(
                    input_tokens=raw_usage.get("prompt_tokens", 0),
                    output_tokens=raw_usage.get("completion_tokens", 0),
                )
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
                if on_delta is not None:
                    on_delta(delta["content"])
            for tc in delta.get("tool_calls") or []:
                frag = tool_fragments.setdefault(
                    tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    frag["id"] += tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    frag["name"] += function["name"]
                if function.get("arguments"):
                    frag["arguments"] += function["arguments"]
        calls = [
            ToolCall(
                id=frag["id"] or f"call_{i}",
                name=frag["name"],
                arguments=_parse_arguments(frag["arguments"]),
            )
            for i, frag in sorted(tool_fragments.items())
        ]
        return ProviderResponse(
            message=Message.assistant("".join(text_parts), calls),
            stop_reason=finish_reason,
            usage=usage,
            model=model,
        )


class GroqProvider(OpenAIProvider):
    """Groq exposes an OpenAI-compatible Chat Completions endpoint."""

    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "llama-3.3-70b-versatile"
    api_key_env = "GROQ_API_KEY"
