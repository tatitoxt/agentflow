from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agentflow import Message, ProviderResponse, ToolCall, ToolSpec
from agentflow.errors import ConfigurationError, ProviderError
from agentflow.providers import (
    AnthropicProvider,
    FakeProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    provider_from_env,
)

SPEC = ToolSpec(
    name="get_weather",
    description="Weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
CONVERSATION = [
    Message.system("be brief"),
    Message.user("weather in Lima?"),
    Message.assistant(
        tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "Lima"})]
    ),
    Message.tool("c1", "get_weather", "18C"),
]


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def body(self, index: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


def openai_reply(**message: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


def test_fake_provider_replays_script_and_records_calls() -> None:
    provider = FakeProvider(
        [
            FakeProvider.tool_call("get_weather", city="Lima"),
            lambda messages, tools: f"last tool said {messages[-1].content}",
        ]
    )
    first = provider.complete([Message.user("hi")], [SPEC])
    assert first.stop_reason == "tool_use"
    assert first.message.tool_calls[0].arguments == {"city": "Lima"}
    second = provider.complete([Message.tool("call_get_weather", "get_weather", "18C")], [])
    assert second.message.content == "last tool said 18C"
    assert second.stop_reason == "end_turn" and second.model == "fake-1"
    assert len(provider.calls) == 2
    with pytest.raises(ProviderError, match="script exhausted"):
        provider.complete([], [])


def test_fake_provider_accepts_single_responder() -> None:
    provider = FakeProvider(
        lambda messages, tools: ProviderResponse(message=Message.assistant("x"))
    )
    assert provider.complete([], []).message.content == "x"
    assert provider.complete([], []).message.content == "x"


def test_openai_wire_format_and_parsing() -> None:
    reply = openai_reply(
        content=None,
        tool_calls=[
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Quito"}'},
            }
        ],
    )
    rec = Recorder(reply)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-test", client=rec.client())
    response = provider.complete(CONVERSATION, [SPEC])

    request = rec.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = rec.body()
    assert body["model"] == "gpt-test" and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Lima"}',
    }
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "18C"}
    assert body["tools"][0]["function"]["name"] == "get_weather"

    assert response.message.tool_calls == [
        ToolCall(id="c2", name="get_weather", arguments={"city": "Quito"})
    ]
    assert response.usage.input_tokens == 11 and response.usage.output_tokens == 3


def test_openai_malformed_arguments_are_preserved_for_validation() -> None:
    reply = openai_reply(
        tool_calls=[{"id": "c", "function": {"name": "get_weather", "arguments": "{oops"}}]
    )
    provider = OpenAIProvider(api_key="k", client=Recorder(reply).client())
    call = provider.complete([Message.user("x")], [SPEC]).message.tool_calls[0]
    assert call.arguments == {"_raw_arguments": "{oops"}


def test_groq_uses_openai_compatible_endpoint() -> None:
    rec = Recorder(openai_reply(content="hola"))
    provider = GroqProvider(api_key="gsk-test", client=rec.client())
    assert provider.complete([Message.user("hi")], []).message.content == "hola"
    assert str(rec.requests[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert rec.body()["model"] == "llama-3.3-70b-versatile"
    assert "tools" not in rec.body()


def test_anthropic_wire_format_and_parsing() -> None:
    rec = Recorder(
        httpx.Response(
            200,
            json={
                "model": "claude-test",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "get_weather",
                        "input": {"city": "Bogota"},
                    },
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )
    )
    provider = AnthropicProvider(api_key="ak-test", model="claude-test", client=rec.client())
    conversation = [*CONVERSATION, Message.tool("c9", "get_weather", "boom", is_error=True)]
    response = provider.complete(conversation, [SPEC])

    request = rec.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "ak-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = rec.body()
    assert body["system"] == "be brief" and body["max_tokens"] == 1024
    assert body["tools"][0]["input_schema"] == SPEC.parameters
    assert body["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Lima"}}
        ],
    }
    # consecutive tool results are merged into one user turn
    results = body["messages"][2]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["c1", "c9"]
    assert results["content"][1]["is_error"] is True

    assert response.message.content == "Checking."
    assert response.message.tool_calls[0].arguments == {"city": "Bogota"}
    assert response.stop_reason == "tool_use" and response.usage.output_tokens == 5


def test_retries_on_retryable_status_then_succeeds() -> None:
    rec = Recorder(httpx.Response(429, text="slow down"), openai_reply(content="ok"))
    provider = OpenAIProvider(api_key="k", client=rec.client(), backoff=0)
    assert provider.complete([Message.user("x")], []).message.content == "ok"
    assert len(rec.requests) == 2


def test_gives_up_after_max_retries() -> None:
    rec = Recorder(*(httpx.Response(503, text="down") for _ in range(3)))
    provider = OpenAIProvider(api_key="k", client=rec.client(), backoff=0, max_retries=2)
    with pytest.raises(ProviderError) as info:
        provider.complete([Message.user("x")], [])
    assert info.value.status_code == 503 and len(rec.requests) == 3


def test_client_errors_are_not_retried() -> None:
    rec = Recorder(httpx.Response(401, text="bad key"))
    provider = AnthropicProvider(api_key="k", client=rec.client(), backoff=0)
    with pytest.raises(ProviderError, match="HTTP 401"):
        provider.complete([Message.user("x")], [])
    assert len(rec.requests) == 1


def test_transport_errors_become_provider_errors() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = httpx.Client(transport=httpx.MockTransport(fail))
    provider = OpenAIProvider(api_key="k", client=client, backoff=0, max_retries=1)
    with pytest.raises(ProviderError, match="request failed"):
        provider.complete([Message.user("x")], [])


def test_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProvider(["hi"])
    for var in ("AGENTFLOW_PROVIDER", "OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert provider_from_env(fake=fake) is fake
    with pytest.raises(ConfigurationError, match="no script"):
        provider_from_env()
    monkeypatch.setenv("AGENTFLOW_PROVIDER", "openai")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        provider_from_env(fake=fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AGENTFLOW_MODEL", "gpt-custom")
    provider = provider_from_env(fake=fake)
    assert isinstance(provider, OpenAIProvider) and provider.model == "gpt-custom"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    assert isinstance(provider_from_env("anthropic"), AnthropicProvider)
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    assert isinstance(provider_from_env("groq"), GroqProvider)
    assert isinstance(provider_from_env("ollama"), OllamaProvider)
    with pytest.raises(ConfigurationError, match="unknown provider"):
        provider_from_env("mystery")


def test_ollama_needs_no_key_and_hits_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OLLAMA_BASE_URL", "AGENTFLOW_MODEL"):
        monkeypatch.delenv(var, raising=False)
    provider = OllamaProvider.from_env()
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "llama3.2"

    recorder = Recorder(openai_reply(content="hola"))
    provider = OllamaProvider(client=recorder.client())
    response = provider.complete([Message.user("hi")], [])
    assert response.message.content == "hola"
    assert recorder.requests[0].url == "http://localhost:11434/v1/chat/completions"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    monkeypatch.setenv("AGENTFLOW_MODEL", "qwen3")
    remote = OllamaProvider.from_env()
    assert remote.base_url == "http://gpu-box:11434/v1"
    assert remote.model == "qwen3"


def sse(*events: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_openai_stream_reassembles_text_and_tool_calls() -> None:
    reply = sse(
        {"model": "gpt-test", "choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "get_weather", "arguments": '{"cit'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'y": "Lima"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
    )
    provider = OpenAIProvider(api_key="k", client=Recorder(reply).client())
    deltas: list[str] = []
    response = provider.stream([Message.user("x")], [SPEC], deltas.append)
    assert deltas == ["Hel", "lo"]
    assert response.message.content == "Hello"
    assert response.message.tool_calls == [
        ToolCall(id="c1", name="get_weather", arguments={"city": "Lima"})
    ]
    assert response.stop_reason == "tool_calls"
    assert response.usage.input_tokens == 9 and response.usage.output_tokens == 4


def test_openai_stream_sends_stream_options() -> None:
    rec = Recorder(sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))
    provider = OpenAIProvider(api_key="k", client=rec.client())
    provider.stream([Message.user("x")], [])
    body = rec.body()
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_anthropic_stream_reassembles_blocks() -> None:
    reply = sse(
        {
            "type": "message_start",
            "message": {"model": "claude-test", "usage": {"input_tokens": 7}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "there"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"Quito"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 12},
        },
    )
    provider = AnthropicProvider(api_key="k", client=Recorder(reply).client())
    deltas: list[str] = []
    response = provider.stream([Message.user("x")], [SPEC], deltas.append)
    assert deltas == ["Hi ", "there"]
    assert response.message.content == "Hi there"
    assert response.message.tool_calls == [
        ToolCall(id="t1", name="get_weather", arguments={"city": "Quito"})
    ]
    assert response.stop_reason == "tool_use"
    assert response.usage.input_tokens == 7 and response.usage.output_tokens == 12
