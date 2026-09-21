from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from analysis.report import build_phase1_report
from benchmark.loader import discover_tasks, select_tasks
from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest
from runner.runner import ExperimentRunner

app = typer.Typer(no_args_is_help=True)
console = Console()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_runtime(config: ExperimentConfig) -> tuple[dict, dict, dict | None]:
    root = config.root
    agents = load_yaml(root / "configs/agents.yaml")["agents"]
    models = load_yaml(root / "configs/models.yaml")["models"]
    model_manifest = None
    for model_name in config.experiment.get("models", []):
        model_config = models.get(model_name, {})
        if model_config.get("provider") == "university_gpu" or model_name == "university_gpu":
            endpoint, model_manifest = ensure_model_manifest(
                root, config.inference, verify_tool=True
            )
            models[model_name] = endpoint.model_config()
    return agents, models, model_manifest


@app.command("list-tasks")
def list_tasks(tasks_dir: Path = typer.Option(Path("benchmark/tasks"), exists=True)) -> None:
    table = Table("Task", "Category", "Source")
    for task in discover_tasks(tasks_dir):
        table.add_row(task.task_id, task.category, task.source_task or "")
    console.print(table)


@app.command("run")
def run(
    config: Path = typer.Option(Path("configs/phase1.research.yaml"), exists=True),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_cells: int | None = typer.Option(None, min=1),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    agents, models, _ = load_runtime(experiment_config)
    task_ids = experiment_config.configured_task_ids
    task_defs = select_tasks(experiment_config.task_root, task_ids)
    runner = ExperimentRunner(experiment_config, agents, models)
    try:
        matrix = runner.validate_matrix(task_defs)
        results = runner.run(task_defs, resume=resume, max_cells=max_cells)
        summary = runner.completion_summary(task_defs)
    finally:
        runner.close()
    successful = sum(1 for result in results if result["status"] == "completed" and result["success"])
    completed = sum(1 for result in results if result["status"] == "completed")
    console.print(
        f"Matrix {matrix['cells']} cells; processed {len(results)}; "
        f"completed {completed}, successful {successful}, "
        f"infrastructure errors {summary['infrastructure_error']}, pending {summary['pending']}."
    )
    if max_cells is None and (
        summary["gradable"] != summary["expected"] or summary["infrastructure_error"]
    ):
        raise typer.Exit(code=2)


@app.command("report")
def report(
    database: Path = typer.Option(Path("data/phase1/experiment/runs.sqlite"), exists=True),
    output: Path = typer.Option(Path("data/phase1/experiment/provisional_report.md")),
) -> None:
    build_phase1_report(database, output)
    console.print(f"Wrote {output}")


if __name__ == "__main__":
    app()
