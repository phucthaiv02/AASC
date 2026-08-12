from __future__ import annotations

import json
from types import SimpleNamespace

from aicomp_sdk.agents.types import (
    AgentToolSpec,
    ToolCall,
    ToolCallDecision,
    ToolResult,
)
from aicomp_sdk.core.runtime_history import RuntimeHistory

from aas_nim_validation.nim_agent import NIMChatCompletionsAgent


class FakeResponse:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]

    def model_dump(self, **_kwargs):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


def test_agent_maps_chat_tool_call_and_history():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="fs_read", arguments=json.dumps({"path": "x.txt"})),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    completions = FakeCompletions(FakeResponse(message))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = NIMChatCompletionsAgent(client=client, model="test/model")
    tools = (
        AgentToolSpec(
            name="fs.read",
            description="Read a file",
            parameters_json_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    )
    history = RuntimeHistory().with_instruction("Use tools").with_user_message("Read x.txt")

    decision = agent.next_action(history=history, tools=tools)

    assert isinstance(decision, ToolCallDecision)
    assert decision.call.tool_name == "fs.read"
    assert decision.call.arguments == {"path": "x.txt"}
    assert completions.request["parallel_tool_calls"] is False
    assert completions.request["messages"][0] == {"role": "system", "content": "Use tools"}

    continued = history.with_tool_request(
        ToolCall(call_id="call-1", tool_name="fs.read", arguments={"path": "x.txt"})
    ).with_tool_result(
        ToolResult(call_id="call-1", tool_name="fs.read", output_text="hello")
    )
    rendered = agent._render_messages(continued, {"fs.read": "fs_read"})
    assert rendered[-2]["tool_calls"][0]["function"]["name"] == "fs_read"
    assert rendered[-1]["role"] == "tool"
    assert rendered[-1]["content"] == "hello"

