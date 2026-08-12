from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from aicomp_sdk.agents.debug import (
    AgentDebugEvent,
    AgentDebugSink,
    serialize_agent_decision,
    summarize_runtime_history,
)
from aicomp_sdk.agents.protocol import AgentProtocol
from aicomp_sdk.agents.tool_specs import build_openai_tool_name_maps
from aicomp_sdk.agents.types import (
    AGENT_STATE_VERSION,
    AgentDecision,
    AgentStateSnapshot,
    AgentStateVersionError,
    AgentToolSpec,
    AssistantMessageEvent,
    FinalResponseDecision,
    InvalidModelOutputError,
    ToolCall,
    ToolCallDecision,
    ToolRequestEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from aicomp_sdk.core.runtime_history import RuntimeHistory


class NIMChatCompletionsAgent(AgentProtocol):
    """AAS agent adapter for NIM's OpenAI-compatible Chat Completions API."""

    _BACKEND = "nim_chat_completions"

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        debug_sink: AgentDebugSink | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._debug_sink = debug_sink
        self._next_turn_index = 1

    def next_action(
        self,
        *,
        history: RuntimeHistory,
        tools: Sequence[AgentToolSpec],
    ) -> AgentDecision:
        canonical_to_nim, nim_to_canonical = build_openai_tool_name_maps(tools)
        request = self._build_request(history, tools, canonical_to_nim)
        turn_index = self._next_turn_index
        self._record(
            "request_built", turn_index, history, request_payload=request,
            provider_payload={"tool_names": list(canonical_to_nim.values())},
        )

        started_at = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        response_payload = response.model_dump(mode="json", exclude_unset=True)
        self._record(
            "response_received", turn_index, history,
            response_payload=response_payload, latency_ms=latency_ms,
        )

        try:
            decision = self._parse_response(response, nim_to_canonical)
        except InvalidModelOutputError as error:
            self._record(
                "parse_error", turn_index, history,
                request_payload=request, response_payload=response_payload,
                error=str(error), latency_ms=latency_ms,
            )
            raise

        self._record(
            "decision_emitted", turn_index, history,
            decision_payload=serialize_agent_decision(decision), latency_ms=latency_ms,
        )
        self._next_turn_index += 1
        return decision

    def reset_state(self) -> None:
        self._next_turn_index = 1

    def snapshot_state(self) -> AgentStateSnapshot:
        return {
            "version": AGENT_STATE_VERSION,
            "backend": self._BACKEND,
            "data": {"next_turn_index": self._next_turn_index},
        }

    def restore_state(self, snapshot: AgentStateSnapshot) -> None:
        if snapshot.get("version") != AGENT_STATE_VERSION:
            raise AgentStateVersionError(f"Unsupported agent snapshot version: {snapshot.get('version')}")
        if snapshot.get("backend") != self._BACKEND:
            raise AgentStateVersionError(f"Unsupported agent snapshot backend: {snapshot.get('backend')}")
        value = snapshot.get("data", {}).get("next_turn_index")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AgentStateVersionError("Invalid next_turn_index in agent snapshot")
        self._next_turn_index = value

    def _build_request(
        self,
        history: RuntimeHistory,
        tools: Sequence[AgentToolSpec],
        canonical_to_nim: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": self._render_messages(history, canonical_to_nim),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": canonical_to_nim.get(spec.name, spec.name),
                        "description": spec.description,
                        "parameters": dict(spec.parameters_json_schema),
                    },
                }
                for spec in tools
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    @staticmethod
    def _render_messages(
        history: RuntimeHistory,
        canonical_to_nim: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        instructions = "\n\n".join(
            event.text for event in history.instructions if event.text.strip()
        )
        if instructions:
            messages.append({"role": "system", "content": instructions})

        for event in history.events:
            if isinstance(event, UserMessageEvent):
                messages.append({"role": "user", "content": event.text})
            elif isinstance(event, AssistantMessageEvent):
                messages.append({"role": "assistant", "content": event.text})
            elif isinstance(event, ToolRequestEvent):
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": event.call.call_id,
                                "type": "function",
                                "function": {
                                    "name": canonical_to_nim.get(
                                        event.call.tool_name, event.call.tool_name
                                    ),
                                    "arguments": json.dumps(
                                        event.call.arguments, ensure_ascii=False, sort_keys=True
                                    ),
                                },
                            }
                        ],
                    }
                )
            elif isinstance(event, ToolResultEvent):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.result.call_id,
                        "name": canonical_to_nim.get(
                            event.result.tool_name, event.result.tool_name
                        ),
                        "content": event.result.output_text,
                    }
                )
            else:
                raise InvalidModelOutputError(f"Unsupported runtime event: {event!r}")
        return messages

    @staticmethod
    def _parse_response(response: Any, nim_to_canonical: Mapping[str, str]) -> AgentDecision:
        if not response.choices:
            raise InvalidModelOutputError("NIM response contains no choices")
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        content = (message.content or "").strip()

        if len(tool_calls) > 1:
            raise InvalidModelOutputError("NIM returned multiple tool calls; AAS expects one action")
        if tool_calls:
            tool_call = tool_calls[0]
            try:
                arguments = json.loads(tool_call.function.arguments)
            except (TypeError, json.JSONDecodeError) as error:
                raise InvalidModelOutputError("Tool call arguments are not valid JSON") from error
            if not isinstance(arguments, dict):
                raise InvalidModelOutputError("Tool call arguments must decode to an object")
            raw_name = tool_call.function.name
            return ToolCallDecision(
                call=ToolCall(
                    call_id=tool_call.id,
                    tool_name=nim_to_canonical.get(raw_name, raw_name),
                    arguments=arguments,
                ),
                assistant_message=content or None,
            )
        if content:
            return FinalResponseDecision(text=content)
        raise InvalidModelOutputError("NIM produced neither assistant text nor a tool call")

    def _record(
        self,
        phase: str,
        turn_index: int,
        history: RuntimeHistory,
        *,
        request_payload: Mapping[str, Any] | None = None,
        response_payload: Mapping[str, Any] | None = None,
        decision_payload: Mapping[str, Any] | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
        provider_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._debug_sink is None:
            return
        self._debug_sink.record(
            AgentDebugEvent(
                backend=self._BACKEND,
                model=self._model,
                phase=phase,
                turn_index=turn_index,
                history_summary=summarize_runtime_history(history),
                request_payload=request_payload,
                response_payload=response_payload,
                decision_payload=decision_payload,
                error=error,
                latency_ms=latency_ms,
                provider_payload=provider_payload or {},
            )
        )

