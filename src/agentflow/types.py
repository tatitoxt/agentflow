"""Core message and response types shared by agents, tools and providers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single chat message in provider-neutral form."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, name: str, content: str, is_error: bool = False) -> Message:
        return cls(
            role="tool", tool_call_id=tool_call_id, name=name, content=content, is_error=is_error
        )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderResponse(BaseModel):
    """What a provider returns for one completion request."""

    message: Message
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None


class ToolSpec(BaseModel):
    """Provider-neutral description of a tool: name, description and JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
