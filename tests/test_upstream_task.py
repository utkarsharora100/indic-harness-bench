import shutil
import sqlite3
from pathlib import Path

import runner.runner as runner_module
from agents.base import AgentResponse
from benchmark.models import TaskDefinition
from runner.config import ExperimentConfig
from runner.runner import ExperimentRunner


class LineCountAgent:
    name = "linecount"

    def __init__(self) -> None:
        self.instructions: list[str] = []

    def run(self, request):
        self.instructions.append(request.instruction)
        (Path(request.workspace) / "out" / "linecount.txt").write_text("4\n", encoding="utf-8")
        return AgentResponse(completed=True, text="done")

    def close(self) -> None:
        return None


def test_runner_executes_copied_upstream_task_in_all_languages(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "project"
    source_task = Path("benchmark/tasks/001-file")
    destination_task = root / "benchmark/tasks/001-file"
    destination_task.parent.mkdir(parents=True)
    shutil.copytree(source_task, destination_task)

    config = ExperimentConfig(
        root / "configs/phase1.yaml",
        {
            "experiment": {
                "seed": 17,
                "repetitions": 1,
                "languages": ["english", "hindi", "hinglish"],
                "agents": ["react"],
                "models": ["local"],
            },
            "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 100},
            "sandbox": {"mode": "local", "image": "python:3.12-slim"},
            "storage": {
                "database": "data/runs.sqlite",
                "traces_dir": "data/traces",
                "results_dir": "data/results",
            },
        },
    )
    agent = LineCountAgent()
    monkeypatch.setattr(runner_module, "build_agent", lambda *args, **kwargs: agent)
    task = TaskDefinition.from_file(destination_task / "task.yaml")
    runner = ExperimentRunner(config, {"react": {"type": "react"}}, {"local": {"model": "test"}})
    try:
        results = runner.run([task])
    finally:
        runner.close()

    assert len(results) == 3
    assert all(result["success"] for result in results)
    assert any("कार्य निर्देशिका" in instruction for instruction in agent.instructions)
    assert any("Working directory" in instruction for instruction in agent.instructions)
    with sqlite3.connect(root / "data/runs.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM grade WHERE passed = 1").fetchone()[0] == 3
