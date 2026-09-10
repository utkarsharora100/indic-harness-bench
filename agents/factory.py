from __future__ import annotations

import os
from typing import Any

from agents.base import AgentAdapter
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
        api_key = os.environ.get(api_key_name, "ollama")
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

    raise ValueError(f"Unsupported agent type: {kind}")
