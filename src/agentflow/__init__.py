"""AgentFlow: a lightweight, typed framework for tool-calling AI agents."""

from .agent import (
    Agent,
    RunResult,
    RunState,
    StopCondition,
    cost_budget,
    max_tool_calls,
    stop_after_tool,
    token_budget,
)
from .errors import (
    AgentFlowError,
    ConfigurationError,
    ProviderError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)
from .pricing import PRICES, estimate_cost, register_price
from .providers.base import Provider, StreamingProvider
from .tools import Tool, ToolRegistry, ToolResult, tool
from .tracing import Event, JSONLSink, MemorySink, Tracer, read_jsonl
from .types import Message, ProviderResponse, ToolCall, ToolSpec, Usage

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Event",
    "JSONLSink",
    "MemorySink",
    "RunResult",
    "RunState",
    "Provider",
    "StopCondition",
    "StreamingProvider",
    "Tracer",
    "PRICES",
    "cost_budget",
    "estimate_cost",
    "max_tool_calls",
    "register_price",
    "read_jsonl",
    "stop_after_tool",
    "token_budget",
    "AgentFlowError",
    "ConfigurationError",
    "Message",
    "ProviderError",
    "ProviderResponse",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "Usage",
    "__version__",
    "tool",
]
