from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from agents.base import AgentAdapter, AgentRequest, AgentResponse


@dataclass(slots=True)
class ExternalAgentConfig:
    command: str
    args: list[str]
    message_args: list[str]
    message_file_args: list[str]
    timeout_seconds: int


class ExternalAgentAdapter:
    def __init__(self, name: str, config: ExternalAgentConfig) -> None:
        self.name = name
        self.config = config

    def run(self, request: AgentRequest) -> AgentResponse:
        workspace = Path(request.workspace)
        message_file = workspace / ".benchmark_message.txt"
        message_file.write_text(request.instruction, encoding="utf-8")

        command = [self.config.command, *self.config.args]
        if self.config.message_file_args:
            command.extend([*self.config.message_file_args, str(message_file)])
        else:
            command.extend([*self.config.message_args, request.instruction])

        started = monotonic()
        try:
            process = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=os.environ.copy(),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Agent executable not found: {self.config.command}") from exc
        except subprocess.TimeoutExpired as exc:
            return AgentResponse(
                completed=False,
                text=exc.stdout or "",
                metadata={"timeout": True},
            )

        metadata: dict[str, object] = {
            "returncode": process.returncode,
            "stderr": process.stderr[-12000:],
            "elapsed_seconds": monotonic() - started,
        }

        if self.name == "openclaw":
            metadata["openclaw"] = _parse_openclaw_json(process.stdout)

        return AgentResponse(
            completed=process.returncode == 0,
            text=process.stdout,
            metadata=metadata,
        )

    def close(self) -> None:
        return None


def _parse_openclaw_json(stdout: str) -> dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"json_parse_error": True}
    return data if isinstance(data, dict) else {"json_parse_error": True}
