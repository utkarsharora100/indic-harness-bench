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
                                    "maxTokens": 2048,
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
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            elif self.name == "openclaw":
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
                                                "maxTokens": 2048,
                                            }
                                        ],
                                    }
                                }
                            },
                            "agents": {
                                "defaults": {
                                    "workspace": "/workspace",
                                    "model": {"primary": f"phase1/{request.model}"},
                                },
                                "entries": {
                                    "main": {
                                        "workspace": "/workspace",
                                        "model": {"primary": f"phase1/{request.model}", "fallbacks": []},
                                        "thinkingDefault": "off",
                                        "params": {"temperature": 0.0, "maxTokens": 2048},
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
                "PHASE1_MAX_TOKENS": "2048",
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
                extra_hosts={"host.docker.internal": "host-gateway"},
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
                completed=returncode == 0 and not timed_out,
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
