from __future__ import annotations

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from agentflow import Tool, ToolCall, ToolRegistry, tool
from agentflow.errors import ToolExecutionError, ToolValidationError


@tool
def add(a: int, b: int = 2) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend,
            defaults to two.
    """
    return a + b


class Ticket(BaseModel):
    title: str
    priority: Literal["low", "high"] = "low"


@tool(name="file_ticket", description="Create a ticket")
def create_ticket(ticket: Ticket) -> dict[str, str]:
    return {"title": ticket.title, "priority": ticket.priority}


@tool
def boom(reason: str) -> str:
    """Always fails."""
    raise RuntimeError(reason)


def test_schema_from_type_hints_and_docstring() -> None:
    spec = add.spec
    assert spec.name == "add"
    assert spec.description == "Add two integers."
    assert spec.parameters["type"] == "object"
    assert spec.parameters["required"] == ["a"]
    assert spec.parameters["properties"]["a"] == {"type": "integer", "description": "First addend."}
    assert spec.parameters["properties"]["b"]["default"] == 2
    assert spec.parameters["properties"]["b"]["description"] == "Second addend, defaults to two."
    assert spec.parameters["additionalProperties"] is False


def test_annotated_field_description() -> None:
    @tool
    def search(query: Annotated[str, Field(description="Search text", min_length=1)]) -> str:
        return query

    prop = search.spec.parameters["properties"]["query"]
    assert prop["description"] == "Search text"
    assert prop["minLength"] == 1


def test_pydantic_model_parameter_uses_model_schema() -> None:
    spec = create_ticket.spec
    assert spec.name == "file_ticket"
    assert spec.description == "Create a ticket"
    assert spec.parameters["properties"]["priority"]["enum"] == ["low", "high"]
    assert create_ticket({"title": "x", "priority": "high"}) == {"title": "x", "priority": "high"}


def test_call_validates_and_coerces() -> None:
    assert add({"a": 1}) == 3
    assert add({"a": "4", "b": 1}) == 5


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({}, "a: Field required"),
        ({"a": "not-a-number"}, "a: Input should be a valid integer"),
        ({"a": 1, "c": 3}, "c: Extra inputs are not permitted"),
    ],
)
def test_invalid_arguments_raise_validation_error(
    arguments: dict[str, object], fragment: str
) -> None:
    with pytest.raises(ToolValidationError) as info:
        add(arguments)
    assert fragment in str(info.value)


def test_exceptions_are_wrapped() -> None:
    with pytest.raises(ToolExecutionError, match="RuntimeError: nope"):
        boom({"reason": "nope"})


def test_missing_type_hint_rejected() -> None:
    def untyped(x):  # type: ignore[no-untyped-def]
        return x

    with pytest.raises(TypeError, match="needs a type hint"):
        Tool(untyped)


def test_invalid_name_rejected() -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        Tool(add.fn, name="has spaces")


def test_registry_execute_never_raises() -> None:
    registry = ToolRegistry([add, boom])
    ok = registry.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}))
    assert (ok.content, ok.is_error, ok.value) == ("2", False, 2)

    bad_args = registry.execute(ToolCall(id="2", name="add", arguments={"a": "x"}))
    assert bad_args.is_error and isinstance(bad_args.error, ToolValidationError)
    assert bad_args.content.startswith("ERROR: invalid arguments for tool 'add'")

    failed = registry.execute(ToolCall(id="3", name="boom", arguments={"reason": "r"}))
    assert failed.is_error and isinstance(failed.error, ToolExecutionError)

    missing = registry.execute(ToolCall(id="4", name="nope", arguments={}))
    assert missing.is_error and "unknown tool: 'nope'" in missing.content


def test_registry_serializes_results_and_rejects_duplicates() -> None:
    registry = ToolRegistry([create_ticket])
    result = registry.execute(ToolCall(id="1", name="file_ticket", arguments={"title": "t"}))
    assert result.content == '{"title": "t", "priority": "low"}'
    with pytest.raises(ValueError, match="already registered"):
        registry.register(create_ticket)
    assert "file_ticket" in registry and len(registry) == 1
    assert [s.name for s in registry.specs()] == ["file_ticket"]


def test_plain_function_registration() -> None:
    def shout(text: str) -> str:
        """Upper-case text."""
        return text.upper()

    registry = ToolRegistry([shout])
    assert registry.get("shout").description == "Upper-case text."
    assert (
        registry.execute(ToolCall(id="1", name="shout", arguments={"text": "hi"})).content == "HI"
    )
