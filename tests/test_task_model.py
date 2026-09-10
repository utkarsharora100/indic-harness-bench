from pathlib import Path

from benchmark.models import TaskDefinition


def test_demo_task_loads():
    task = TaskDefinition.from_file(Path("benchmark/tasks/demo/task.yaml"))
    assert task.task_id == "demo"
    assert task.instruction_for("english")
    assert task.instruction_for("hindi")
    assert task.instruction_for("hinglish")
