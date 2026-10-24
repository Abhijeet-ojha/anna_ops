from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from annaops.config import Settings, settings
from annaops.llm import LLMInvocation, StructuredLLMClient, TestMockClient
from annaops.schemas import AgentDecision


class NoArguments(BaseModel):
    pass


class AgentProtocolError(RuntimeError):
    """Model proposed an invalid action, unknown tool, or exhausted iterations."""
    category = "agent_protocol_failure"


class AgentExecutionLimitError(AgentProtocolError):
    """Agent reached the maximum iteration limit without a valid final decision."""
    category = "agent_execution_limit_failure"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[[BaseModel], Any]
    arguments_model: type[BaseModel] = NoArguments


@dataclass(frozen=True)
class RuntimeOutcome:
    decision: AgentDecision
    tool_events: list[dict[str, Any]]
    invocations: list[LLMInvocation]


class BoundedToolRuntime:
    """Runs a model/tool loop with a hard iteration cap and backend-owned tools."""

    def __init__(self, config: Settings = settings, client: StructuredLLMClient | None = None):
        self.config = config
        if client is None and getattr(config, "testing_mode", False):
            self.client = TestMockClient(config)
        else:
            self.client = client or StructuredLLMClient(config)

    def run(
        self,
        *,
        agent_name: str,
        objective: str,
        context: dict[str, Any],
        tools: list[ToolSpec],
        allowed_actions: set[str],
        required_tools: set[str] | None = None,
    ) -> RuntimeOutcome:
        registry = {tool.name: tool for tool in tools}
        required_tools = required_tools or set()
        tool_events: list[dict[str, Any]] = []
        invocations: list[LLMInvocation] = []
        executed: set[str] = set()
        tool_catalog = [
            {
                "name": tool.name,
                "description": tool.description,
                "arguments_schema": tool.arguments_model.model_json_schema(),
            }
            for tool in tools
        ]
        for iteration in range(1, self.config.agent_max_iterations + 1):
            invocation = self.client.complete_structured(
                AgentDecision,
                system_prompt=(
                    f"You are the {agent_name}. Select only registered tools and allowed final actions. "
                    "When calling tools, set proposed_action='use_tools' and populate tool_calls. "
                    "Only choose from allowed_final_actions when concluding your final decision. "
                    "Use tools before deciding when current evidence is insufficient. Never claim a tool result "
                    "you have not received. Keep evidence concise and do not reveal chain-of-thought."
                ),
                user_payload={
                    "objective": objective,
                    "context": context,
                    "available_tools": tool_catalog,
                    "allowed_final_actions": sorted(allowed_actions),
                    "required_tools_before_final_action": sorted(required_tools),
                    "tool_results_so_far": tool_events,
                    "iteration": iteration,
                    "iterations_remaining": self.config.agent_max_iterations - iteration,
                },
            )
            decision = AgentDecision.model_validate(invocation.output)
            invocations.append(invocation)
            if decision.tool_calls:
                if decision.proposed_action != "use_tools":
                    decision.proposed_action = "use_tools"
                for requested in decision.tool_calls:
                    spec = registry.get(requested.name)
                    if spec is None:
                        raise AgentProtocolError(f"Model requested unknown tool '{requested.name}'")
                    try:
                        arguments = spec.arguments_model.model_validate(requested.arguments)
                    except ValidationError as exc:
                        raise AgentProtocolError(f"Invalid arguments for tool '{requested.name}'") from exc
                    result = spec.handler(arguments)
                    tool_events.append({
                        "sequence": len(tool_events) + 1,
                        "name": spec.name,
                        "arguments": arguments.model_dump(mode="json"),
                        "purpose": requested.purpose,
                        "result": result,
                    })
                    executed.add(spec.name)
                continue
            if decision.proposed_action not in allowed_actions:
                raise AgentProtocolError(f"Model proposed forbidden action '{decision.proposed_action}'")
            missing_tools = required_tools - executed
            if missing_tools:
                raise AgentProtocolError(f"Model decided before required tools ran: {sorted(missing_tools)}")
            return RuntimeOutcome(decision=decision, tool_events=tool_events, invocations=invocations)
        raise AgentExecutionLimitError("Agent iteration limit reached before a valid terminal decision")
