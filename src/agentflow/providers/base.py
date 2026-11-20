"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import time
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

    def close(self) -> None:
        self._client.close()
