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
