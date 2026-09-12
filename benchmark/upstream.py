"""Compatibility helpers for copied Harness-Bench task assets.

The original repository remains read-only.  Imported task assets live below an
Indic task's ``source/`` directory, while this project owns the language
overlays and execution orchestration.
"""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from pathlib import Path
from typing import Any


def copy_fixtures(fixtures: Path, workspace: Path) -> None:
    """Prepare the same ``in/`` and ``out/`` workspace convention as upstream."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "in").mkdir(exist_ok=True)
    (workspace / "out").mkdir(exist_ok=True)
    if not fixtures.is_dir():
        return
    for child in fixtures.iterdir():
        destination = workspace / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def render_instruction(template: str, workspace: Path | str) -> str:
    """Render the only runtime placeholder supported by upstream prompts."""
    return template.replace("$WORKSPACE", str(workspace))


def run_oracle(task_dir: Path, oracle_module: str, workspace: Path) -> dict[str, Any]:
    """Run an upstream ``score_workspace(workspace)`` function without importing its package."""
    oracle_path = task_dir / oracle_module
    if not oracle_path.is_file():
        return {"outcome_score": 0.0, "error": f"missing oracle module: {oracle_module}"}

    module_name = f"indic_harness_oracle_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, oracle_path)
    if spec is None or spec.loader is None:
        return {"outcome_score": 0.0, "error": f"unable to load oracle module: {oracle_module}"}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        scorer = getattr(module, "score_workspace", None)
        if not callable(scorer):
            return {"outcome_score": 0.0, "error": "oracle missing score_workspace(workspace)"}
        result = scorer(workspace)
        if not isinstance(result, dict):
            return {"outcome_score": 0.0, "error": "oracle returned a non-object result"}
        return result
    except Exception as exc:
        return {"outcome_score": 0.0, "error": f"oracle failed: {type(exc).__name__}: {exc}"}
