from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from agents.base import AgentAdapter, AgentRequest, AgentResponse


@dataclass(slots=True)
class ContainerHarnessConfig:
    image: str
    command: list[str]
    network: str = "none"
    timeout_seconds: int = 600


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
        message_file = workspace / ".benchmark_message.txt"
        message_file.write_text(request.instruction, encoding="utf-8")
        values = {
            "workspace": "/workspace",
            "message_file": "/workspace/.benchmark_message.txt",
            "prompt": request.instruction,
            "model": request.model,
        }
        command = [part.format(**values) for part in self.config.command]
        proxy_url = str(self.model_config.get("container_base_url") or self.model_config.get("base_url", ""))
        env = {
            "OPENAI_BASE_URL": proxy_url,
            "OPENAI_API_BASE": proxy_url,
            "OPENAI_API_KEY": str(self.model_config.get("api_key", "phase1")),
            "OPENAI_MODEL": request.model,
            "PHASE1_MODEL": request.model,
            "WORKSPACE": "/workspace",
        }
        client = docker.from_env()
        container = None
        started = monotonic()
        try:
            client.images.get(self.config.image)
            container = client.containers.run(
                self.config.image,
                command=command,
                working_dir="/workspace",
                environment=env,
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                network_disabled=self.config.network == "none",
                network=self.config.network if self.config.network not in {"none", ""} else None,
                detach=True,
            )
            result = container.wait(timeout=max(self.config.timeout_seconds, request.command_timeout_seconds or 0, 30))
            output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            returncode = int(result.get("StatusCode", 1)) if isinstance(result, dict) else 1
            return AgentResponse(
                completed=returncode == 0,
                text=output,
                metadata={
                    "returncode": returncode,
                    "elapsed_seconds": monotonic() - started,
                    "container_image": self.config.image,
                    "native_harness_output": output[-12000:],
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
                client.close()

    def close(self) -> None:
        return None
