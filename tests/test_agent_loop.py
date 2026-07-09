from types import SimpleNamespace

import pytest

from arxiv_curator.llm.agent_loop import AgentLoopError, ToolSpec, run_tool_loop

FINALIZE_TOOL = ToolSpec(
    name="finalize", description="Finalize decisions.",
    parameters_json_schema={
        "type": "object",
        "properties": {"decisions": {"type": "array"}},
        "required": ["decisions"],
    },
    handler=None,
)


def function_call_part(name, args):
    return SimpleNamespace(function_call=SimpleNamespace(name=name, args=args))


def response_with_parts(parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


class ScriptedModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        return self._responses.pop(0)


class ScriptedClient:
    def __init__(self, responses):
        self.models = ScriptedModels(responses)


def test_run_tool_loop_returns_finalize_args():
    client = ScriptedClient([
        response_with_parts([function_call_part(
            "finalize", {"decisions": [{"arxiv_id": "p1", "status": "picked", "reasoning": "great"}]},
        )]),
    ])
    result = run_tool_loop(client, "system prompt", tools=[FINALIZE_TOOL])
    assert result == {"decisions": [{"arxiv_id": "p1", "status": "picked", "reasoning": "great"}]}


def test_run_tool_loop_calls_handler_and_continues():
    calls = []

    def handler(arxiv_id):
        calls.append(arxiv_id)
        return {"title": "Some Paper"}

    detail_tool = ToolSpec(
        name="get_paper_detail", description="Get detail.",
        parameters_json_schema={
            "type": "object", "properties": {"arxiv_id": {"type": "string"}},
            "required": ["arxiv_id"],
        },
        handler=handler,
    )
    client = ScriptedClient([
        response_with_parts([function_call_part("get_paper_detail", {"arxiv_id": "p1"})]),
        response_with_parts([function_call_part("finalize", {"decisions": []})]),
    ])

    result = run_tool_loop(client, "system prompt", tools=[detail_tool, FINALIZE_TOOL])

    assert calls == ["p1"]
    assert result == {"decisions": []}
    second_call_contents = client.models.calls[1]["contents"]
    last_part = second_call_contents[-1].parts[0]
    assert last_part.function_response.name == "get_paper_detail"


def test_run_tool_loop_raises_when_call_cap_exceeded():
    def handler(arxiv_id):
        return {}

    detail_tool = ToolSpec(
        name="get_paper_detail", description="Get detail.",
        parameters_json_schema={
            "type": "object", "properties": {"arxiv_id": {"type": "string"}},
            "required": ["arxiv_id"],
        },
        handler=handler,
    )
    responses = [
        response_with_parts([function_call_part("get_paper_detail", {"arxiv_id": "p1"})])
        for _ in range(3)
    ]
    client = ScriptedClient(responses)

    with pytest.raises(AgentLoopError, match="Exceeded"):
        run_tool_loop(client, "system prompt", tools=[detail_tool, FINALIZE_TOOL], max_tool_calls=2)


def test_run_tool_loop_raises_when_model_returns_no_tool_call():
    client = ScriptedClient([
        response_with_parts([SimpleNamespace(function_call=None)]),
    ])
    with pytest.raises(AgentLoopError, match="no tool call"):
        run_tool_loop(client, "system prompt", tools=[FINALIZE_TOOL])
