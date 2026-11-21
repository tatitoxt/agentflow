"""Anthropic Messages API adapter (Claude)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
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

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        payload = self.build_payload(messages, tools)
        payload["stream"] = True
        text_parts: list[str] = []
        tool_blocks: dict[int, dict[str, str]] = {}
        input_tokens, output_tokens = 0, 0
        stop_reason: str | None = None
        model: str | None = None
        for event in self._stream_events("/v1/messages", payload):
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") or {}
                model = model or message.get("model")
                input_tokens = (message.get("usage") or {}).get("input_tokens", 0)
            elif kind == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tool_blocks[event["index"]] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": "",
                    }
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                index = event.get("index", 0)
                if delta.get("type") == "text_delta" and delta.get("text"):
                    text_parts.append(delta["text"])
                    if on_delta is not None:
                        on_delta(delta["text"])
                elif delta.get("type") == "input_json_delta" and index in tool_blocks:
                    tool_blocks[index]["arguments"] += delta.get("partial_json", "")
            elif kind == "message_delta":
                delta = event.get("delta") or {}
                stop_reason = delta.get("stop_reason") or stop_reason
                output_tokens = (event.get("usage") or {}).get("output_tokens", output_tokens)
        calls = []
        for index in sorted(tool_blocks):
            block = tool_blocks[index]
            try:
                arguments = json.loads(block["arguments"]) if block["arguments"] else {}
            except ValueError:
                arguments = {"_raw_arguments": block["arguments"]}
            calls.append(
                ToolCall(
                    id=block["id"] or f"toolu_{index}",
                    name=block["name"],
                    arguments=arguments if isinstance(arguments, dict) else {"_raw": arguments},
                )
            )
        return ProviderResponse(
            message=Message.assistant("".join(text_parts), calls),
            stop_reason=stop_reason,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=model,
        )
