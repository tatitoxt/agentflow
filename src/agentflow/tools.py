"""Tool registry: turn typed Python functions into validated, schema-described tools."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .errors import ToolExecutionError, ToolNotFoundError, ToolValidationError
from .types import ToolCall, ToolSpec

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    sections = re.split(r"\n\s*\n", doc)
    summary = " ".join(sections[0].split())
    params: dict[str, str] = {}
    match = re.search(r"^(?:Args|Arguments|Parameters):\s*\n(?P<body>(?:[ \t]+.+\n?)+)", doc, re.M)
    if match:
        current: str | None = None
        for line in match.group("body").splitlines():
            item = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
            if item:
                current = item.group(1)
                params[current] = item.group(2).strip()
            elif current and line.strip():
                params[current] = f"{params[current]} {line.strip()}"
    return summary, params


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str, ensure_ascii=False)


@dataclass
class ToolResult:
    """Outcome of executing a tool call. `content` is what the model sees."""

    call: ToolCall
    content: str
    is_error: bool = False
    error: Exception | None = None
    value: Any = None


class Tool:
    """A callable plus the pydantic model that validates its arguments."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid tool name {self.name!r}: use [a-zA-Z0-9_-]{{1,64}}")
        summary, param_docs = _parse_docstring(fn.__doc__)
        self.description = description or summary or self.name
        self._model_param: str | None = None
        self._positional: list[str] = []
        self.args_model = self._build_model(param_docs)

    def _build_model(self, param_docs: dict[str, str]) -> type[BaseModel]:
        signature = inspect.signature(self.fn)
        hints = get_type_hints(self.fn, include_extras=True)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        if len(params) == 1:
            annotation = hints.get(params[0].name)
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                self._model_param = params[0].name
                if "extra" in annotation.model_config:
                    return annotation
                strict: type[BaseModel] = type(
                    annotation.__name__,
                    (annotation,),
                    {
                        "model_config": ConfigDict(extra="forbid"),
                        "__module__": annotation.__module__,
                    },
                )
                return strict
        self._positional = [p.name for p in params if p.kind is p.POSITIONAL_ONLY]
        fields: dict[str, Any] = {}
        for param in params:
            if param.name not in hints:
                raise TypeError(f"tool {self.name!r}: parameter {param.name!r} needs a type hint")
            default = ... if param.default is inspect.Parameter.empty else param.default
            doc = param_docs.get(param.name)
            fields[param.name] = (
                hints[param.name],
                Field(default, description=doc) if doc else default,
            )
        model: type[BaseModel] = create_model(
            f"{self.name}_arguments",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return model

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.args_model.model_json_schema()),
        )

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(arguments)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolValidationError(self.name, details) from exc

    def __call__(self, arguments: dict[str, Any]) -> Any:
        """Validate `arguments` and invoke the function. Raises ToolError subclasses."""
        validated = self.validate(arguments)
        try:
            if self._model_param is not None:
                return self.fn(**{self._model_param: validated})
            values = {k: getattr(validated, k) for k in type(validated).model_fields}
            args = [values.pop(k) for k in self._positional]
            return self.fn(*args, **values)
        except Exception as exc:
            raise ToolExecutionError(self.name, exc) from exc


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a typed function into a :class:`Tool`.

    Usable as ``@tool`` or ``@tool(name="...", description="...")``.
    """
    if fn is not None:
        return Tool(fn)

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description)

    return wrap


class ToolRegistry:
    """Named collection of tools that executes tool calls and never raises by default."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool | Callable[..., Any]) -> Tool:
        resolved = t if isinstance(t, Tool) else Tool(t)
        if resolved.name in self._tools:
            raise ValueError(f"tool {resolved.name!r} is already registered")
        self._tools[resolved.name] = resolved
        return resolved

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            value = self.get(call.name)(call.arguments)
            try:
                content = _serialize(value)
            except Exception as exc:
                raise ToolExecutionError(call.name, exc) from exc
        except (ToolNotFoundError, ToolValidationError, ToolExecutionError) as exc:
            return ToolResult(call=call, content=f"ERROR: {exc}", is_error=True, error=exc)
        return ToolResult(call=call, content=content, value=value)
