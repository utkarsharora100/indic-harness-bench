from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from analysis.report import build_phase1_report
from analysis.corrected_report import build_corrected_report
from benchmark.loader import discover_tasks, select_tasks
from runner.judge import judge_database
from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest
from runner.runner import ExperimentRunner
from runner.preflight import PreflightError, full_preflight

app = typer.Typer(no_args_is_help=True)
console = Console()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_runtime(config: ExperimentConfig) -> tuple[dict, dict, dict | None]:
    root = config.root
    agents = load_yaml(config.agents_manifest)["agents"]
    models = load_yaml(config.models_manifest)["models"]
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
    try:
        gate = full_preflight(experiment_config, task_defs, agents, models)
    except PreflightError as exc:
        console.print(f"Preflight failed: {exc}")
        raise typer.Exit(code=2) from exc
    console.print(
        f"Preflight passed: {gate['matrix']['cells']} cells, "
        f"{gate['prepared_tasks']} prepared tasks and {gate['language_variants']} variants."
    )
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


@app.command("corrected-report")
def corrected_report(
    config: Path = typer.Option(Path("configs/phase1.corrected.research.yaml"), exists=True),
    database: Path | None = typer.Option(None),
    output: Path = typer.Option(Path("data/phase1/corrected/provisional_report.md")),
    experiment_id: str | None = typer.Option(None),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    db = database or (experiment_config.root / experiment_config.storage["database"])
    result = build_corrected_report(db, output, experiment_id or experiment_config.experiment_id)
    console.print(
        f"Wrote {output}; {result['cells']['completed_rows']} completed cells, "
        f"{result['cells']['oracle_gradable']} oracle-gradable."
    )


@app.command("judge")
def judge(
    config: Path = typer.Option(Path("configs/phase1.corrected.research.yaml"), exists=True),
    database: Path | None = typer.Option(None),
    experiment_id: str | None = typer.Option(None),
    rerun: bool = typer.Option(False, "--rerun", help="Recompute all process judgments under the current frozen normalizer."),
) -> None:
    """Run the frozen paper-style process rubric after agent execution."""
    experiment_config = ExperimentConfig.load(config)
    agents, models, _ = load_runtime(experiment_config)
    runner = ExperimentRunner(experiment_config, agents, models)
    try:
        model_name = str(experiment_config.experiment["models"][0])
        result = judge_database(
            experiment_config.root / (database or experiment_config.storage["database"]),
            experiment_config.task_root,
            runner.models[model_name],
            experiment_id=experiment_id or experiment_config.experiment_id,
            max_tokens=int(experiment_config.generation.get("max_tokens", 2048)),
            proxy=runner.proxies.get(model_name),
            rerun=rerun,
        )
    finally:
        runner.close()
    console.print(result)


if __name__ == "__main__":
    app()
