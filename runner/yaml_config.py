"""Shared, typed YAML loading for tracked study configuration documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlConfigError(ValueError):
    """A configuration file is unreadable or has the wrong top-level shape."""


def load_yaml_mapping(path: Path, *, label: str = "YAML document") -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise YamlConfigError(f"Could not load {label}: {path}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise YamlConfigError(f"{label} must be a YAML mapping: {path}")
    return value
