from dataclasses import dataclass
from typing import Callable, Optional

from google.genai import types

from arxiv_curator.llm.retry import with_retries

DEFAULT_MODEL = "gemini-2.5-flash"


class AgentLoopError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters_json_schema: dict
    handler: Optional[Callable[..., dict]]


def run_tool_loop(
    client,
    system_prompt: str,
    tools: list[ToolSpec],
    finalize_tool_name: str = "finalize",
    max_tool_calls: int = 8,
    model: str = DEFAULT_MODEL,
) -> dict:
    tool_by_name = {t.name: t for t in tools}
    declarations = [
        types.FunctionDeclaration(
            name=t.name, description=t.description,
            parameters_json_schema=t.parameters_json_schema,
        )
        for t in tools
    ]
    config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=declarations)])
    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]

    calls_made = 0
    while True:
        response = with_retries(
            client.models.generate_content, model=model, contents=contents, config=config,
        )
        parts = response.candidates[0].content.parts
        contents.append(response.candidates[0].content)

        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not function_calls:
            raise AgentLoopError("Model returned no tool call and did not finalize.")

        response_parts = []
        for call in function_calls:
            if call.name == finalize_tool_name:
                return dict(call.args)
            calls_made += 1
            if calls_made > max_tool_calls:
                raise AgentLoopError(f"Exceeded {max_tool_calls} tool calls without finalize.")
            spec = tool_by_name[call.name]
            result = spec.handler(**call.args)
            response_parts.append(types.Part.from_function_response(name=call.name, response=result))

        contents.append(types.Content(role="user", parts=response_parts))
