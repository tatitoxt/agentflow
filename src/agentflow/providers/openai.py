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

    async def acomplete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        data = await self._apost("/chat/completions", self.build_payload(messages, tools))
        return from_openai_response(data)

    def _stream_payload(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        payload = self.build_payload(messages, tools)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return payload

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        acc = _StreamAccumulator(on_delta)
        for chunk in self._stream_events(
            "/chat/completions", self._stream_payload(messages, tools)
        ):
            acc.feed(chunk)
        return acc.finish()

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        acc = _StreamAccumulator(on_delta)
        async for chunk in self._astream_events(
            "/chat/completions", self._stream_payload(messages, tools)
        ):
            acc.feed(chunk)
        return acc.finish()


class _StreamAccumulator:
    """Fold Chat Completions stream chunks into a ProviderResponse."""

    def __init__(self, on_delta: Callable[[str], None] | None = None) -> None:
        self._on_delta = on_delta
        self._text_parts: list[str] = []
        self._tool_fragments: dict[int, dict[str, str]] = {}
        self._finish_reason: str | None = None
        self._usage = Usage()
        self._model: str | None = None

    def feed(self, chunk: dict[str, Any]) -> None:
        self._model = self._model or chunk.get("model")
        raw_usage = chunk.get("usage")
        if raw_usage:
            self._usage = Usage(
                input_tokens=raw_usage.get("prompt_tokens", 0),
                output_tokens=raw_usage.get("completion_tokens", 0),
            )
        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        self._finish_reason = choice.get("finish_reason") or self._finish_reason
        delta = choice.get("delta") or {}
        if delta.get("content"):
            self._text_parts.append(delta["content"])
            if self._on_delta is not None:
                self._on_delta(delta["content"])
        for tc in delta.get("tool_calls") or []:
            frag = self._tool_fragments.setdefault(
                tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
            )
            if tc.get("id"):
                frag["id"] += tc["id"]
            function = tc.get("function") or {}
            if function.get("name"):
                frag["name"] += function["name"]
            if function.get("arguments"):
                frag["arguments"] += function["arguments"]

    def finish(self) -> ProviderResponse:
        if self._finish_reason is None:
            raise ProviderError("stream ended before a finish_reason was received")
        calls = [
            ToolCall(
                id=frag["id"] or f"call_{i}",
                name=frag["name"],
                arguments=_parse_arguments(frag["arguments"]),
            )
            for i, frag in sorted(self._tool_fragments.items())
        ]
        return ProviderResponse(
            message=Message.assistant("".join(self._text_parts), calls),
            stop_reason=self._finish_reason,
            usage=self._usage,
            model=self._model,
        )


class GroqProvider(OpenAIProvider):
    """Groq exposes an OpenAI-compatible Chat Completions endpoint."""

    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "llama-3.3-70b-versatile"
    api_key_env = "GROQ_API_KEY"


class OllamaProvider(OpenAIProvider):
    """Ollama's OpenAI-compatible endpoint — local models, no API key needed.

    ``from_env`` reads ``OLLAMA_BASE_URL`` (default ``http://localhost:11434/v1``)
    and ``AGENTFLOW_MODEL``; a key is never required since Ollama serves localhost
    without auth (an Authorization header is still sent and ignored by Ollama).
    """

    name = "ollama"
    default_base_url = "http://localhost:11434/v1"
    default_model = "llama3.2"

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key or "ollama", **kwargs)

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs: Any) -> OllamaProvider:
        return cls(
            model=model or os.environ.get("AGENTFLOW_MODEL"),
            base_url=kwargs.pop("base_url", None)
            or os.environ.get("OLLAMA_BASE_URL")
            or cls.default_base_url,
            **kwargs,
        )
