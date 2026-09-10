from pathlib import Path

from benchmark.loader import discover_tasks, select_tasks


def test_demo_is_discoverable():
    tasks = discover_tasks(Path("benchmark/tasks"))
    assert any(task.task_id == "demo" for task in tasks)


def test_demo_alias_selects_demo():
    tasks = select_tasks(Path("benchmark/tasks"), ["demo"])
    assert [task.task_id for task in tasks] == ["demo"]
