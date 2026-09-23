from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from agents.base import AgentAdapter, AgentRequest, AgentResponse
from agents.tools import WorkspaceTools, serialise_tool_result


@dataclass(slots=True)
class ReactConfig:
    base_url: str
    api_key: str
    timeout_seconds: int


def _is_malformed_tool_call_error(error: Exception) -> bool:
    """Recognise provider-side JSON parsing failures as model/tool errors."""
    message = str(error).lower()
    return (
        "failed to parse tool call arguments as json" in message
        or "tool call arguments" in message and "parse_error" in message
    )


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
            Path(request.workspace),
            request.command_timeout_seconds or request.max_steps,
            request.command_runner,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.instruction},
        ]

        input_tokens: int | None = None
        output_tokens: int | None = None
        response_text = ""
        events: list[dict[str, Any]] = []
        tool_calls = 0
        failed_tool_calls = 0
        model_calls = 0
        initial_prompt_tokens: int | None = None
        truncation_retries = 0
        task_deadline = monotonic() + (request.command_timeout_seconds or request.max_steps)

        def timed_out_response() -> AgentResponse:
            events.append({"step": model_calls, "event_type": "agent_timeout"})
            return AgentResponse(
                completed=False,
                text=response_text,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None else None,
                },
                metadata={
                    "stop_reason": "task_timeout", "model_calls": model_calls,
                    "initial_prompt_tokens": initial_prompt_tokens,
                    "tool_calls": tool_calls, "failed_tool_calls": failed_tool_calls,
                    "events": events,
                },
            )

        def prepare_shorter_retry() -> None:
            nonlocal messages, truncation_retries
            truncation_retries += 1
            tail = list(messages[2:][-12:])
            while tail and tail[0].get("role") == "tool":
                tail.pop(0)
            messages = [messages[0], messages[1], *tail]
            messages.append({
                "role": "user",
                "content": (
                    "The previous model response reached the generation limit. "
                    "Continue the task using short, valid tool calls. Split long files "
                    "or scripts into small pieces and verify each written artifact."
                ),
            })
            events.append({"step": model_calls, "event_type": "generation_limit_recovery"})

        for step in range(request.max_steps):
            remaining = task_deadline - monotonic()
            if remaining <= 0:
                return timed_out_response()
            model_calls += 1
            try:
                completion = self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    tools=tools.schemas(),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    timeout=remaining,
                )
            except Exception as exc:
                if monotonic() >= task_deadline:
                    return timed_out_response()
                events.append({
                    "step": step + 1,
                    "event_type": "model_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                if _is_malformed_tool_call_error(exc):
                    return AgentResponse(
                        completed=False,
                        text=response_text,
                        usage={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": (
                                input_tokens + output_tokens
                                if input_tokens is not None and output_tokens is not None
                                else None
                            ),
                        },
                        metadata={
                            "stop_reason": "malformed_tool_call",
                            "model_calls": model_calls,
                            "model_tool_errors": 1,
                            "initial_prompt_tokens": initial_prompt_tokens,
                            "tool_calls": tool_calls,
                            "failed_tool_calls": failed_tool_calls,
                            "events": events,
                        },
                    )
                raise

            if not getattr(completion, "choices", None):
                error = ValueError("Model response contained no completion choices")
                events.append({
                    "step": step + 1,
                    "event_type": "model_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                raise error

            if completion.usage is not None:
                if initial_prompt_tokens is None:
                    initial_prompt_tokens = completion.usage.prompt_tokens
                if completion.usage.prompt_tokens is not None:
                    input_tokens = (input_tokens or 0) + completion.usage.prompt_tokens
                if completion.usage.completion_tokens is not None:
                    output_tokens = (output_tokens or 0) + completion.usage.completion_tokens

            model_event = {
                "step": step + 1,
                "event_type": "model_call",
                "finish_reason": getattr(completion.choices[0], "finish_reason", None),
                "usage": {
                    "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
                    "output_tokens": completion.usage.completion_tokens if completion.usage else None,
                    "total_tokens": (
                        (completion.usage.prompt_tokens or 0)
                        + (completion.usage.completion_tokens or 0)
                    ) if completion.usage else None,
                },
            }

            message = completion.choices[0].message
            response_text = message.content or ""
            message_tool_calls = message.tool_calls or []
            finish_reason = getattr(completion.choices[0], "finish_reason", None)

            if not message_tool_calls:
                events.append(model_event)
                if finish_reason == "length" and truncation_retries < 1:
                    prepare_shorter_retry()
                    continue
                return AgentResponse(
                    # A response cut off at the per-call generation budget is
                    # a valid, gradable model stop.  Do not resend the same
                    # growing history and turn it into an opaque upstream
                    # context-window failure.
                    completed=finish_reason != "length",
                    text=response_text,
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": (
                            input_tokens + output_tokens
                            if input_tokens is not None and output_tokens is not None
                            else None
                        ),
                    },
                    metadata={
                        "steps": step + 1,
                        "tool_calls": tool_calls,
                        "failed_tool_calls": failed_tool_calls,
                        "model_calls": model_calls,
                        "initial_prompt_tokens": initial_prompt_tokens,
                        "stop_reason": "max_tokens" if finish_reason == "length" else None,
                        "events": events,
                    },
                )

            message_start = len(messages)
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
                remaining = task_deadline - monotonic()
                if remaining <= 0:
                    return timed_out_response()
                tools.timeout_seconds = max(1, int(remaining))
                tool_calls += 1
                tool_name = call.function.name
                raw_arguments = call.function.arguments
                arguments: dict[str, Any] = {}
                result: dict[str, Any]
                tool_failed = False
                try:
                    decoded = json.loads(raw_arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    arguments = decoded
                    result = tools.execute(tool_name, arguments)
                except Exception as exc:
                    failed_tool_calls += 1
                    tool_failed = True
                    result = {
                        "ok": False,
                        "error": str(exc),
                        "type": type(exc).__name__,
                        "raw_arguments": raw_arguments,
                    }

                if result.get("ok") is False and not tool_failed:
                    failed_tool_calls += 1

                events.append({
                    "step": step + 1,
                    "event_type": "tool_call",
                    "tool": tool_name,
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

            # Keep tool observations first for backwards-compatible traces;
            # the model-call event still records per-call usage and finish
            # reason without collecting private reasoning text.
            events.append(model_event)

            if finish_reason == "length":
                if truncation_retries < 1:
                    # A truncated JSON tool call cannot be replayed as a
                    # valid assistant/tool exchange on the next model call.
                    if any(event.get("step") == step + 1 and event.get("result", {}).get("type") == "JSONDecodeError"
                           for event in events if event.get("event_type") == "tool_call"):
                        del messages[message_start:]
                    prepare_shorter_retry()
                    continue
                # The final tool call is valid and has been executed above.
                # Stop here rather than issuing a continuation that can push
                # the accumulated tool transcript beyond the served model's
                # context window.
                return AgentResponse(
                    completed=False,
                    text=response_text,
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": (
                            input_tokens + output_tokens
                            if input_tokens is not None and output_tokens is not None
                            else None
                        ),
                    },
                    metadata={
                        "steps": step + 1,
                        "stop_reason": "max_tokens",
                        "tool_calls": tool_calls,
                        "failed_tool_calls": failed_tool_calls,
                        "model_calls": model_calls,
                        "initial_prompt_tokens": initial_prompt_tokens,
                        "events": events,
                    },
                )
            truncation_retries = 0

        return AgentResponse(
            completed=False,
            text=response_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
            },
            metadata={
                "stop_reason": "max_steps",
                "steps": request.max_steps,
                "tool_calls": tool_calls,
                "failed_tool_calls": failed_tool_calls,
                "model_calls": model_calls,
                "initial_prompt_tokens": initial_prompt_tokens,
                "events": events,
            },
        )

    def close(self) -> None:
        self.client.close()


def build_react_agent(base_url: str, api_key: str, timeout_seconds: int) -> AgentAdapter:
    return ReactAgent(ReactConfig(base_url, api_key, timeout_seconds))

