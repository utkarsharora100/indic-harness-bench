from __future__ import annotations

from pathlib import Path

from benchmark.models import TaskDefinition


def discover_tasks(root: Path) -> list[TaskDefinition]:
    return [
        TaskDefinition.from_file(path)
        for path in sorted(root.glob("*/task.yaml"))
    ]


def select_tasks(root: Path, task_ids: list[str]) -> list[TaskDefinition]:
    tasks = discover_tasks(root)
    by_id = {task.task_id: task for task in tasks}

    if task_ids == ["demo"]:
        return [task for task in tasks if task.source_task == "local_demo"]

    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Unknown task IDs: {', '.join(missing)}")

    return [by_id[task_id] for task_id in task_ids]
