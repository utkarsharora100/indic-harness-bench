from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(slots=True)
class AgentRequest:
    instruction: str
    workspace: str
    system_prompt: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    max_steps: int
    command_runner: Callable[[str], dict[str, Any]] | None = None


@dataclass(slots=True)
class AgentResponse:
    completed: bool
    text: str
    usage: dict[str, int | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(Protocol):
    name: str

    def run(self, request: AgentRequest) -> AgentResponse:
        ...

    def close(self) -> None:
        ...
