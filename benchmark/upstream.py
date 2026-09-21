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
from types import SimpleNamespace
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
    """Render Harness-Bench workspace and hook variables in a prompt.

    The upstream runner substitutes ``$WORKSPACE`` and variables returned by
    task hooks before handing the prompt to the harness.  Keeping this logic in
    one place prevents the translated prompt path from diverging from the
    shell/file-tool workspace.
    """
    return render_runtime_template(template, workspace=workspace)


def render_runtime_template(
    template: str,
    *,
    workspace: Path | str,
    runtime_env: dict[str, str] | None = None,
) -> str:
    values = {"WORKSPACE": str(workspace)}
    values.update({key: str(value) for key, value in (runtime_env or {}).items()})
    rendered = template
    for key in sorted(values, key=len, reverse=True):
        rendered = rendered.replace(f"${key}", values[key])
    return rendered


def _task_context(task_dir: Path) -> Any:
    """Provide the task attributes used by the upstream hook contract."""
    task_yaml = task_dir / "task.yaml"
    return SimpleNamespace(
        task_dir=task_dir,
        task_id=task_dir.name,
        task_yaml=task_yaml,
    )


def load_task_module(task_dir: Path, module_name: str) -> Any | None:
    path = task_dir / module_name
    if not path.is_file():
        return None
    name = f"indic_harness_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_runtime(task_dir: Path, sandbox: Path, workspace: Path, hooks_module: str) -> dict[str, Any]:
    """Run the pinned task hook exactly once before an agent starts.

    Hooks are benchmark assets, not model code.  They are run by the
    orchestrator against the fresh host-mounted workspace, which is the same
    directory later mounted at ``/workspace`` in the agent container.
    """
    module = load_task_module(task_dir, hooks_module)
    if module is None:
        return {}
    prepare = getattr(module, "prepare_runtime", None)
    if not callable(prepare):
        return {}
    state = prepare(
        {"task": _task_context(task_dir), "sandbox": sandbox, "workspace": workspace}
    )
    return dict(state) if isinstance(state, dict) else {}


def after_round_runtime(
    task_dir: Path,
    sandbox: Path,
    workspace: Path,
    hooks_module: str,
    state: dict[str, Any],
    adapter_result: Any,
    round_index: int = 0,
) -> dict[str, Any]:
    """Run the upstream per-round hook and merge its returned state."""
    module = load_task_module(task_dir, hooks_module)
    after_round = getattr(module, "after_round", None) if module is not None else None
    if not callable(after_round):
        return state
    result = after_round(
        {
            "task": _task_context(task_dir),
            "sandbox": sandbox,
            "workspace": workspace,
            "round_index": round_index,
        },
        state,
        adapter_result,
    )
    if isinstance(result, dict):
        state.update(result)
    return state


def cleanup_runtime(task_dir: Path, sandbox: Path, workspace: Path, hooks_module: str, state: dict[str, Any]) -> None:
    module = load_task_module(task_dir, hooks_module)
    cleanup = getattr(module, "cleanup_runtime", None) if module is not None else None
    if callable(cleanup):
        cleanup(
            {"task": _task_context(task_dir), "sandbox": sandbox, "workspace": workspace},
            state,
        )


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
