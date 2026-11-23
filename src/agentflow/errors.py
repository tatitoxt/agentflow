"""Exception hierarchy for AgentFlow."""

from __future__ import annotations


class AgentFlowError(Exception):
    """Base class for all AgentFlow errors."""


class ToolError(AgentFlowError):
    """A tool could not be executed."""


class ToolNotFoundError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown tool: {name!r}")
        self.name = name


class ToolValidationError(ToolError):
    def __init__(self, name: str, details: str) -> None:
        super().__init__(f"invalid arguments for tool {name!r}: {details}")
        self.name = name
        self.details = details


class ToolExecutionError(ToolError):
    def __init__(self, name: str, cause: BaseException) -> None:
        super().__init__(f"tool {name!r} raised {type(cause).__name__}: {cause}")
        self.name = name
        self.cause = cause


class ProviderError(AgentFlowError):
    """A provider request failed (network, HTTP status or malformed response)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StructuredOutputError(AgentFlowError):
    """The agent's final answer could not be parsed into the requested schema,
    even after the configured number of repair turns."""

    def __init__(self, message: str, *, output: str, errors: str) -> None:
        super().__init__(message)
        self.output = output
        self.errors = errors


class ConfigurationError(AgentFlowError):
    """Missing or invalid configuration (for example an API key env var)."""
