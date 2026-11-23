"""Recording/replay provider fixtures."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentflow.errors import ProviderError
from agentflow.providers.fake import FakeProvider
from agentflow.providers.replay import RecordingProvider, ReplayProvider
from agentflow.types import Message, ProviderResponse, ToolSpec


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(message=Message.assistant(text))


def test_record_then_replay_roundtrip(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    inner = FakeProvider([_resp("hello"), _resp("world")])
    rec = RecordingProvider(inner, path)
    msgs = [Message.user("hi")]
    assert rec.complete(msgs, []).message.content == "hello"
    assert rec.complete(msgs, []).message.content == "world"

    replay = ReplayProvider(path)
    assert replay.complete(msgs, []).message.content == "hello"
    assert replay.complete(msgs, []).message.content == "world"


def test_replay_exhausted_raises(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingProvider(FakeProvider([_resp("one")]), path).complete([Message.user("q")], [])
    replay = ReplayProvider(path)
    replay.complete([Message.user("q")], [])
    with pytest.raises(ProviderError, match="exhausted"):
        replay.complete([Message.user("q")], [])


def test_replay_request_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingProvider(FakeProvider([_resp("one")]), path).complete([Message.user("first")], [])
    replay = ReplayProvider(path)
    with pytest.raises(ProviderError, match="does not match"):
        replay.complete([Message.user("different")], [])


def test_replay_op_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingProvider(FakeProvider([_resp("one")]), path).complete([Message.user("q")], [])
    replay = ReplayProvider(path)
    with pytest.raises(ProviderError, match="diverged"):
        replay.stream([Message.user("q")], [])


def test_stream_replay_invokes_on_delta(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    inner = FakeProvider([_resp("streamed")])
    RecordingProvider(inner, path).stream([Message.user("q")], [], lambda d: None)
    got: list[str] = []
    replay = ReplayProvider(path)
    resp = replay.stream([Message.user("q")], [], got.append)
    assert resp.message.content == "streamed" and got == ["streamed"]


def test_async_roundtrip(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"

    async def run() -> None:
        inner = FakeProvider([_resp("async one")])
        rec = RecordingProvider(inner, path)
        await rec.acomplete([Message.user("q")], [])
        replay = ReplayProvider(path)
        assert (await replay.acomplete([Message.user("q")], [])).message.content == "async one"

    asyncio.run(run())


def test_fixture_records_tool_calls(tmp_path) -> None:
    path = tmp_path / "fixture.jsonl"
    inner = FakeProvider([FakeProvider.tool_call("lookup", q="x")])
    rec = RecordingProvider(inner, path)
    tools = [ToolSpec(name="lookup", description="d", parameters={"type": "object"})]
    resp = rec.complete([Message.user("find x")], tools)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["request"]["tools"][0]["name"] == "lookup"
    assert resp.message.tool_calls[0].name == "lookup"


def test_provider_from_env_wiring(tmp_path, monkeypatch) -> None:
    from agentflow.providers import provider_from_env

    fixture = tmp_path / "env.jsonl"
    RecordingProvider(FakeProvider([_resp("env")]), fixture).complete([Message.user("q")], [])
    monkeypatch.setenv("AGENTFLOW_PROVIDER", "replay")
    monkeypatch.setenv("AGENTFLOW_REPLAY_FILE", str(fixture))
    p = provider_from_env()
    assert p.name == "replay"
    assert p.complete([Message.user("q")], []).message.content == "env"

    # AGENTFLOW_RECORD_FILE wraps whatever provider was built.
    monkeypatch.setenv("AGENTFLOW_PROVIDER", "fake")
    monkeypatch.setenv("AGENTFLOW_RECORD_FILE", str(tmp_path / "rec.jsonl"))
    p = provider_from_env(fake=FakeProvider([_resp("recorded")]))
    p.complete([Message.user("q")], [])
    assert (tmp_path / "rec.jsonl").read_text().strip()
