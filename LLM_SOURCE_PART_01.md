# Indic-Harness-Bench Complete Source Corpus — Part 01

Original branch: prishiv_dev · Commit: f16a2233a2769f06598c02960845135ba5b3e5d8
Export branch: llm-export-prishiv-dev-20261010

Every file below is embedded in full. Original branch unchanged.

## FILE: .github/workflows/test.yml

Git blob SHA: `465141802c696e10ed225fa376a384ac803fe2f3` · Bytes: 319

```yaml
name: test

on:
  push:
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install -e ".[dev]"
      - run: pytest
      - run: python -m compileall -q .

```

---

## FILE: .gitignore

Git blob SHA: `a689c48990169b94bb2b904833efd01d6943d787` · Bytes: 1001

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
.env
.env.*
dist/
build/
*.egg-info/

data/*.sqlite
data/results/*
data/traces/*
!data/results/.gitkeep
!data/traces/.gitkeep
data/phase1/
/output/pdf/indic_harness_phase1_corrected_pilot_v14.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v1.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v2.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v3.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v4.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v5.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v6.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v7.pdf
/output/pdf/indic_harness_phase1_llm_judged_regrade_v8.pdf
/output/pdf/indic_harness_phase1_main24_v1.pdf
/output/pdf/indic_harness_phase1_main24_llm_v5.pdf
benchmark/translations/phase1.partial.yaml

tmp/

benchmark/tasks/*/workspace/*
benchmark/tasks/*/runs/*
!benchmark/tasks/demo/workspace/
!benchmark/tasks/demo/workspace/app.py

```

---

## FILE: .vscode/settings.json

Git blob SHA: `c62197f0ba8f5e11e6e22e65af4590dd7e1f5b9c` · Bytes: 67

```json
{
    "python-envs.defaultPackageManager": "ms-python.python:pip"
}
```

---

## FILE: agents/__init__.py

Git blob SHA: `ef1a1eb8436f9fd45432143cd489460a11719f13` · Bytes: 249

```python
from .base import AgentAdapter, AgentRequest, AgentResponse
from .external import ExternalAgentAdapter

__all__ = ["AgentAdapter", "AgentRequest", "AgentResponse", "ExternalAgentAdapter"]

# Gemini integration is intentionally disabled for Phase 1.

```

---

## FILE: agents/base.py

Git blob SHA: `ffc25b54cd8c310bc9d67322253a513657072890` · Bytes: 856

```python
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
    command_timeout_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

```

---

## FILE: agents/container.py

Git blob SHA: `1125a5b1dac598fedfcb415b43450d641007a1d9` · Bytes: 14231

```python
from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from agents.base import AgentAdapter, AgentRequest, AgentResponse
from agents.external import _parse_openclaw_json


def _write_openclaw_retry_settings(state_dir: Path) -> Path:
    """Disable OpenClaw provider/recovery retries for one-attempt studies."""
    # OpenClaw v2026.9.5 resolves the default agent directory under
    # $OPENCLAW_STATE_DIR/agents/main/agent and loads its settings.json there.
    path = state_dir / "agents" / "main" / "agent" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"retry": {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}}}
        ),
        encoding="utf-8",
    )
    return path


@dataclass(slots=True)
class ContainerHarnessConfig:
    image: str
    command: list[str]
    network: str = "none"
    timeout_seconds: int = 600
    state_mount: str = "/state"


class ContainerHarnessAdapter:
    """Run a native harness image against only the mounted benchmark workspace.

    The image and command are pinned in the experiment configuration.  Model
    access is supplied through the configured proxy URL; the university bearer
    key never enters this container.
    """

    def __init__(self, name: str, config: ContainerHarnessConfig, model_config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.model_config = model_config

    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError("Docker Python package is required for native harnesses") from exc

        workspace = Path(request.workspace).resolve()
        client = docker.from_env()
        container = None
        started = monotonic()
        state_dir = Path(tempfile.mkdtemp(prefix="ihb-native-state-"))
        try:
            message_file = state_dir / "prompt.txt"
            message_file.write_text(request.instruction, encoding="utf-8")
            proxy_url = str(
                self.model_config.get("container_base_url")
                or self.model_config.get("base_url", "")
            )
            client_key = str(self.model_config.get("api_key", "phase1"))
            config_path = state_dir / f"{self.name}.json"
            if self.name == "nanobot":
                config_path.write_text(
                    json.dumps(
                        {
                            "providers": {
                                "custom": {"apiKey": client_key, "apiBase": proxy_url}
                            },
                            "modelPresets": {
                                "primary": {
                                    "provider": "custom",
                                    "model": request.model,
                                    "maxTokens": request.max_tokens,
                                    "temperature": 0.0,
                                }
                            },
                            "agents": {
                                "defaults": {
                                    "modelPreset": "primary",
                                    "workspace": "/workspace",
                                    "maxToolIterations": 40,
                                }
                            },
                            "tools": {"restrictToWorkspace": True},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            elif self.name == "openclaw":
                _write_openclaw_retry_settings(state_dir)
                config_path.write_text(
                    json.dumps(
                        {
                            "models": {
                                "providers": {
                                    "phase1": {
                                        "baseUrl": proxy_url,
                                        "apiKey": client_key,
                                        "api": "openai-completions",
                                        "models": [
                                            {
                                                "id": request.model,
                                                "name": "Phase I university model",
                                                "input": ["text"],
                                                "contextWindow": 32768,
                                                "maxTokens": request.max_tokens,
                                            }
                                        ],
                                    }
                                }
                            },
                            "agents": {
                                "defaults": {
                                    "workspace": "/workspace",
                                    "model": {"primary": f"phase1/{request.model}"},
                                    "compaction": {
                                        "mode": "safeguard",
                                        "midTurnPrecheck": {"enabled": True},
                                        "keepRecentTokens": 6000,
                                        "recentTurnsPreserve": 3,
                                        "identifierPolicy": "strict",
                                        "qualityGuard": {"enabled": False, "maxRetries": 0},
                                        "memoryFlush": {"enabled": False},
                                        "maxActiveTranscriptBytes": 40000,
                                    },
                                },
                                "entries": {
                                    "main": {
                                        "workspace": "/workspace",
                                        "model": {"primary": f"phase1/{request.model}", "fallbacks": []},
                                        "thinkingDefault": "off",
                                        "params": {"temperature": 0.0, "maxTokens": request.max_tokens},
                                        "tools": {
                                            # The release treats `allow` as an
                                            # intersection with the selected
                                            # profile, so use coding as the
                                            # native base and explicitly keep
                                            # only the benchmark primitives.
                                            "profile": "coding",
                                            "allow": [
                                                "read",
                                                "write",
                                                "edit",
                                                "apply_patch",
                                                "ls",
                                                "exec",
                                                "process",
                                                "session_status",
                                            ],
                                        },
                                    }
                                },
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            values = {
                "workspace": "/workspace",
                "state": self.config.state_mount,
                "message_file": f"{self.config.state_mount}/prompt.txt",
                "prompt": request.instruction,
                "model": request.model,
                "session": request.metadata.get("session_id", "phase1-session"),
                "config": f"{self.config.state_mount}/{self.name}.json",
                "timeout": str(int(request.command_timeout_seconds or self.config.timeout_seconds)),
            }
            command = [part.format(**values) for part in self.config.command]
            env = {
                "OPENAI_BASE_URL": proxy_url,
                "OPENAI_API_BASE": proxy_url,
                "OPENAI_API_KEY": str(self.model_config.get("api_key", "phase1")),
                "OPENAI_MODEL": request.model,
                "PHASE1_MODEL": request.model,
                "PHASE1_PUBLIC_MODEL": request.model,
                "PHASE1_CELL_ID": request.metadata.get("cell_id", ""),
                "PHASE1_MAX_CALLS": "40",
                "PHASE1_TEMPERATURE": "0",
                "PHASE1_TOP_P": "1",
                "PHASE1_MAX_TOKENS": str(request.max_tokens),
                "WORKSPACE": "/workspace",
                "NANOBOT_HOME": self.config.state_mount,
                "NANOBOT_WORKSPACE": "/workspace",
                "OPENCLAW_STATE_DIR": self.config.state_mount,
                "OPENCLAW_CONFIG_PATH": f"{self.config.state_mount}/openclaw.json",
                "OPENCLAW_AGENT_WORKSPACE_DIR": "/workspace",
            }
            client.images.get(self.config.image)
            container = client.containers.run(
                self.config.image,
                command=command,
                working_dir="/workspace",
                environment=env,
                volumes={
                    str(workspace): {"bind": "/workspace", "mode": "rw"},
                    str(state_dir): {"bind": self.config.state_mount, "mode": "rw"},
                },
                network_disabled=self.config.network == "none",
                network=self.config.network if self.config.network not in {"none", ""} else None,
                detach=True,
            )
            timeout = int(request.command_timeout_seconds or self.config.timeout_seconds)
            try:
                result = container.wait(timeout=max(timeout, 1))
                timed_out = False
            except Exception as exc:
                # Docker SDK raises its timeout exception rather than
                # returning a process status. Preserve this as an observed
                # task timeout so the cell remains gradable, and retain the
                # exact runtime error in metadata.
                container.kill()
                result = {"StatusCode": 124}
                timed_out = True
                timeout_error = f"{type(exc).__name__}: {exc}"
            output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            returncode = int(result.get("StatusCode", 1)) if isinstance(result, dict) else 1
            native_metadata: dict[str, Any] = {}
            if self.name == "openclaw":
                parsed = _parse_openclaw_json(output)
                completion = parsed.get("completion") if isinstance(parsed, dict) else None
                native_stop = completion.get("stopReason") if isinstance(completion, dict) else None
                native_metadata["stop_reason"] = str(native_stop or "unknown")
                summary = parsed.get("toolSummary") if isinstance(parsed, dict) else None
                if isinstance(summary, dict):
                    native_metadata["tool_summary"] = {
                        "calls": int(summary.get("calls", 0) or 0),
                        "failures": int(summary.get("failures", 0) or 0),
                        "tools": [str(value) for value in summary.get("tools", []) if isinstance(value, str)],
                    }
            elif self.name == "nanobot":
                # NanoBot's one-shot CLI emits each native tool action as a
                # separate arrow line. Keep the raw observable output too;
                # this count is used only for the common run accounting fields.
                calls = sum(1 for line in output.splitlines() if line.lstrip().startswith(("↳", "->")))
                native_metadata["tool_summary"] = {"calls": calls, "failures": 0, "tools": []}
            return AgentResponse(
                completed=(returncode == 0 and not timed_out
                           and (self.name != "openclaw" or native_metadata.get("stop_reason") == "stop")),
                text=output,
                usage={},
                metadata={
                    "returncode": returncode,
                    "elapsed_seconds": monotonic() - started,
                    "container_image": self.config.image,
                    "native_harness_output": output[-12000:],
                    "timed_out": timed_out,
                    "timeout_error": timeout_error if timed_out else None,
                    "state_isolated": True,
                    "workspace_mount": "/workspace",
                    "native_tool_calls": native_metadata.get("tool_summary", {}).get("calls", 0),
                    "native_failed_tool_calls": native_metadata.get("tool_summary", {}).get("failures", 0),
                    "native_tool_summary": native_metadata.get("tool_summary", {}),
                    "stop_reason": "timeout" if timed_out else native_metadata.get("stop_reason", "cli_exit"),
                },
            )
        except Exception as exc:
            if container is not None:
                try:
                    container.kill()
                except Exception:
                    pass
            raise RuntimeError(f"Native harness container failed: {type(exc).__name__}: {exc}") from exc
        finally:
            try:
                if container is not None:
                    container.remove(force=True)
            finally:
                try:
                    for path in state_dir.iterdir():
                        if path.is_file() or path.is_symlink():
                            path.unlink()
                        elif path.is_dir():
                            import shutil

                            shutil.rmtree(path)
                    state_dir.rmdir()
                except OSError:
                    pass
                client.close()

    def close(self) -> None:
        return None

```

---

## FILE: agents/external.py

Git blob SHA: `65201d0f7f6fdee4881b337e9817f0bad931ba8e` · Bytes: 3159

```python
from __future__ import annotations

import json
import os
import re
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
    text = stdout.strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    # OpenClaw may prefix its JSON result with human-readable diagnostics and
    # may pretty-print the result across many lines. Select the top-level
    # object carrying the stable tool summary rather than parsing only the
    # final `}` line.
    for match in re.finditer(r"\{", text):
        try:
            data, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("toolSummary" in data or "completion" in data):
            return data
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"json_parse_error": True}
    return data if isinstance(data, dict) else {"json_parse_error": True}

```

---

## FILE: agents/factory.py

Git blob SHA: `3190f4f6a21ffabe9a809aa2765588be1bca0ad0` · Bytes: 1838

```python
from __future__ import annotations

import os
from typing import Any

from agents.base import AgentAdapter
from agents.container import ContainerHarnessAdapter, ContainerHarnessConfig
from agents.external import ExternalAgentAdapter, ExternalAgentConfig
from agents.react import build_react_agent


def build_agent(
    name: str,
    agent_config: dict[str, Any],
    model_config: dict[str, Any],
    timeout_seconds: int,
) -> AgentAdapter:
    kind = agent_config["type"]

    if kind == "react":
        api_key_name = model_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = model_config.get("api_key") or os.environ.get(api_key_name, "ollama")
        return build_react_agent(
            model_config["base_url"],
            api_key,
            timeout_seconds,
        )

    if kind == "external":
        return ExternalAgentAdapter(
            name=name,
            config=ExternalAgentConfig(
                command=agent_config["command"],
                args=agent_config.get("args", []),
                message_args=agent_config.get("message_args", []),
                message_file_args=agent_config.get("message_file_args", []),
                timeout_seconds=timeout_seconds,
            ),
        )

    if kind == "native":
        command = agent_config.get("command") or []
        if isinstance(command, str):
            command = [command]
        return ContainerHarnessAdapter(
            name=name,
            config=ContainerHarnessConfig(
                image=str(agent_config["image"]),
                command=[str(value) for value in command],
                network=str(agent_config.get("network", "none")),
                timeout_seconds=timeout_seconds,
            ),
            model_config=model_config,
        )

    raise ValueError(f"Unsupported agent type: {kind}")

```

---

## FILE: agents/gemini.py

Git blob SHA: `3415184d93f78ffd1bcf869e3a6a0bb0cfa993cb` · Bytes: 706

```python
"""Placeholder for the Gemini API adapter.

The adapter is intentionally disabled in Phase 1. Keep the connection code here so
that enabling Gemini later does not require changing the experiment runner.
"""

# import os
# from google import genai
#
#
# def create_gemini_client() -> genai.Client:
#     """Create a Gemini client from the GEMINI_API_KEY environment variable."""
#     api_key = os.environ["GEMINI_API_KEY"]
#     return genai.Client(api_key=api_key)
#
#
# Example usage once the Gemini condition is enabled:
#
# client = create_gemini_client()
# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Run the benchmark task.",
# )
# print(response.text)

```

---

## FILE: agents/react.py

Git blob SHA: `43979556560e617940f306d9b85e5b5be2a05c3b` · Bytes: 14596

```python
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


```

---

## FILE: agents/tools.py

Git blob SHA: `f18130b3a91456aabf6a419fb73793392963c008` · Bytes: 7276

```python
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories below a path inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path; use '.' for root."}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a file inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search UTF-8 text files below a directory for a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command with the workspace as the current directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


class WorkspaceTools:
    def __init__(self, workspace: Path, timeout_seconds: int, command_runner=None) -> None:
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner

    def schemas(self) -> list[dict[str, Any]]:
        return TOOLS

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "run_command": self._run_command,
        }
        try:
            handler = handlers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        return handler(**arguments)

    def _resolve(self, relative_path: str) -> Path:
        # Harness-Bench prompts conventionally expose the mounted workspace as
        # /workspace.  ReAct receives that same path even though the actual
        # file tools operate on the host-mounted directory, so normalize both
        # spellings before applying the escape check.
        raw = str(relative_path)
        if raw == "/workspace":
            raw = "."
        elif raw.startswith("/workspace/"):
            raw = raw[len("/workspace/") :]
        path = (self.workspace / raw).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError(f"Path is outside workspace: {relative_path}")
        return path

    def _list_files(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": "path does not exist"}
        return {
            "ok": True,
            "entries": [
                {"name": item.name, "type": "directory" if item.is_dir() else "file"}
                for item in sorted(target.iterdir())
            ],
        }

    def _read_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            return {"ok": False, "error": "file does not exist"}
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": "file is not UTF-8 text"}
        return {"ok": True, "content": content}

    def _write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    def _search_files(self, query: str, path: str) -> dict[str, Any]:
        root = self._resolve(path)
        if not root.exists():
            return {"ok": False, "error": "path does not exist"}
        matches = []
        for target in root.rglob("*"):
            if not target.is_file():
                continue
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if query in text:
                matches.append(str(target.relative_to(self.workspace)))
        return {"ok": True, "matches": matches}

    def _run_command(self, command: str) -> dict[str, Any]:
        if self.command_runner is not None:
            try:
                return self.command_runner(command, self.timeout_seconds)
            except TypeError as exc:
                # Preserve compatibility with small test doubles that still
                # expose the original one-argument callback.
                if "positional" not in str(exc) and "argument" not in str(exc):
                    raise
                return self.command_runner(command)

        try:
            process = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=os.environ.copy(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "timeout": True,
                "stdout": (exc.stdout or "")[-12000:],
                "stderr": (exc.stderr or "")[-12000:],
            }
        return {
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-12000:],
            "stderr": process.stderr[-12000:],
        }


def serialise_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)

```

---

## FILE: analysis/__init__.py

Git blob SHA: `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` · Bytes: 0

```python

```

---

