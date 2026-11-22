"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Protocol, runtime_checkable

import httpx

from ..errors import ProviderError
from ..types import Message, ProviderResponse, ToolSpec

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a conversation plus tool specs into the next message."""

    name: str

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse: ...


@runtime_checkable
class StreamingProvider(Provider, Protocol):
    """A provider that can stream response tokens via server-sent events.

    ``stream`` behaves like ``complete`` but invokes ``on_delta`` with each
    text fragment as it arrives, before the full response is assembled.
    """

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse: ...


@runtime_checkable
class AsyncProvider(Protocol):
    """Async counterpart of :class:`Provider`, used by ``Agent.arun``."""

    name: str

    async def acomplete(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> ProviderResponse: ...


@runtime_checkable
class AsyncStreamingProvider(Provider, Protocol):
    """Async streaming counterpart; ``astream`` mirrors ``stream``.

    Inherits ``Provider`` (not ``AsyncProvider``) so a provider may stream
    asynchronously without an async ``complete``; the loop falls back to the
    sync path for non-streaming calls.
    """

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse: ...


class HTTPProvider:
    """Base class for providers that POST JSON over httpx with simple retry/backoff."""

    name = "http"

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        # A caller-supplied client (custom transport, proxy, auth) must see every
        # request, so the agent loop uses the sync methods instead of a default
        # AsyncClient.
        self.prefer_sync = client is not None
        self._timeout = timeout
        self._aclient: httpx.AsyncClient | None = None
        self._headers = headers
        self.max_retries = max_retries
        self.backoff = backoff

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt and self.backoff:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = self._client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"{self.name}: request failed: {exc}")
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = ProviderError(
                    f"{self.name}: HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"{self.name}: HTTP {response.status_code}: {response.text[:500]}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError(f"{self.name}: response is not JSON") from exc
            if not isinstance(data, dict):
                raise ProviderError(f"{self.name}: unexpected response shape")
            return data
        assert last_error is not None
        raise last_error

    def _parse_sse(self, line: str) -> dict[str, Any] | None:
        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            event = json.loads(data)
        except ValueError:
            return None
        if not isinstance(event, dict):
            return None
        if event.get("type") == "error" or "error" in event:
            raise ProviderError(f"{self.name}: stream error: {json.dumps(event)[:500]}")
        return event

    def _stream_status_error(self, response: httpx.Response) -> ProviderError:
        return ProviderError(
            f"{self.name}: HTTP {response.status_code}: {response.text[:500]}",
            status_code=response.status_code,
        )

    def _stream_events(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """POST with ``stream`` and yield each ``data:`` SSE event as parsed JSON.

        Retryable statuses and connection failures are retried like ``_post``
        until the stream opens; an in-band error event raises ProviderError.
        """
        url = f"{self.base_url}{path}"
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt and self.backoff:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                with self._client.stream(
                    "POST", url, json=payload, headers=self._headers
                ) as response:
                    if response.status_code != 200:
                        response.read()
                        last_error = self._stream_status_error(response)
                        if response.status_code in RETRYABLE_STATUS:
                            continue
                        raise last_error
                    for line in response.iter_lines():
                        if (event := self._parse_sse(line)) is not None:
                            yield event
                    return
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"{self.name}: stream failed: {exc}")
                continue
        assert last_error is not None
        raise last_error

    def _get_aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._timeout)
        return self._aclient

    async def _apost(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Async mirror of ``_post``: same retry/backoff and error mapping."""
        url = f"{self.base_url}{path}"
        client = self._get_aclient()
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt and self.backoff:
                await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                response = await client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"{self.name}: request failed: {exc}")
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = ProviderError(
                    f"{self.name}: HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"{self.name}: HTTP {response.status_code}: {response.text[:500]}",
                    status_code=response.status_code,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError(f"{self.name}: response is not JSON") from exc
            if not isinstance(data, dict):
                raise ProviderError(f"{self.name}: unexpected response shape")
            return data
        assert last_error is not None
        raise last_error

    async def _astream_events(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Async mirror of ``_stream_events``: yields each ``data:`` SSE event."""
        url = f"{self.base_url}{path}"
        client = self._get_aclient()
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt and self.backoff:
                await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                async with client.stream(
                    "POST", url, json=payload, headers=self._headers
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        last_error = self._stream_status_error(response)
                        if response.status_code in RETRYABLE_STATUS:
                            continue
                        raise last_error
                    async for line in response.aiter_lines():
                        if (event := self._parse_sse(line)) is not None:
                            yield event
                    return
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"{self.name}: stream failed: {exc}")
                continue
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        """Close the lazily created async client, if one exists."""
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
