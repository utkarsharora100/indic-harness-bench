from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from analysis.report import build_report
from benchmark.loader import discover_tasks, select_tasks
from runner.config import ExperimentConfig
from runner.runner import ExperimentRunner

app = typer.Typer(no_args_is_help=True)
console = Console()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@app.command("list-tasks")
def list_tasks(tasks_dir: Path = typer.Option(Path("benchmark/tasks"), exists=True)) -> None:
    table = Table("Task", "Category", "Source")
    for task in discover_tasks(tasks_dir):
        table.add_row(task.task_id, task.category, task.source_task or "")
    console.print(table)


@app.command("run")
def run(
    config: Path = typer.Option(Path("configs/phase1.yaml"), exists=True),
    tasks: str = typer.Option("demo"),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    root = experiment_config.root
    agents = load_yaml(root / "configs/agents.yaml")["agents"]
    models = load_yaml(root / "configs/models.yaml")["models"]
    task_defs = select_tasks(root / "benchmark/tasks", [item.strip() for item in tasks.split(",")])
    runner = ExperimentRunner(experiment_config, agents, models)
    try:
        results = runner.run(task_defs)
    finally:
        runner.close()
    passed = sum(int(result["success"]) for result in results)
    console.print(f"Completed {len(results)} runs: {passed} successful, {len(results) - passed} failed.")


@app.command("report")
def report(
    database: Path = typer.Option(Path("data/runs.sqlite"), exists=True),
    output: Path = typer.Option(Path("data/results/phase1.md")),
) -> None:
    build_report(database, output)
    console.print(f"Wrote {output}")


if __name__ == "__main__":
    app()
