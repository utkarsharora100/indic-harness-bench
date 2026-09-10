from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.base import AgentAdapter, AgentRequest, AgentResponse
from agents.tools import WorkspaceTools, serialise_tool_result


@dataclass(slots=True)
class ReactConfig:
    base_url: str
    api_key: str
    timeout_seconds: int


class ReactAgent:
    name = "react"

    def __init__(self, config: ReactConfig) -> None:
        self.config = config
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        tools = WorkspaceTools(
            Path(request.workspace), request.max_steps, request.command_runner
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.instruction},
        ]

        input_tokens = 0
        output_tokens = 0
        response_text = ""
        events: list[dict[str, Any]] = []
        tool_calls = 0
        failed_tool_calls = 0

        for step in range(request.max_steps):
            completion = self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                tools=tools.schemas(),
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            if completion.usage is not None:
                input_tokens += completion.usage.prompt_tokens or 0
                output_tokens += completion.usage.completion_tokens or 0

            message = completion.choices[0].message
            response_text = message.content or ""
            message_tool_calls = message.tool_calls or []

            if not message_tool_calls:
                return AgentResponse(
                    completed=True,
                    text=response_text,
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    metadata={
                        "steps": step + 1,
                        "tool_calls": tool_calls,
                        "failed_tool_calls": failed_tool_calls,
                        "events": events,
                    },
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message_tool_calls
                    ],
                }
            )

            for call in message_tool_calls:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                tool_calls += 1
                try:
                    result = tools.execute(call.function.name, arguments)
                except Exception as exc:
                    failed_tool_calls += 1
                    result = {"ok": False, "error": str(exc), "type": type(exc).__name__}

                events.append({
                    "step": step + 1,
                    "event_type": "tool_call",
                    "tool": call.function.name,
                    "arguments": arguments,
                    "result": result,
                })
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": serialise_tool_result(result),
                    }
                )

        return AgentResponse(
            completed=False,
            text=response_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            metadata={
                "stop_reason": "max_steps",
                "steps": request.max_steps,
                "tool_calls": tool_calls,
                "failed_tool_calls": failed_tool_calls,
                "events": events,
            },
        )

    def close(self) -> None:
        self.client.close()


def build_react_agent(base_url: str, api_key: str, timeout_seconds: int) -> AgentAdapter:
    return ReactAgent(ReactConfig(base_url, api_key, timeout_seconds))

