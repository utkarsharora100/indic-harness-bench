from __future__ import annotations

import shutil
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def import_task(
    source: Path = typer.Option(..., exists=True, file_okay=False),
    task: str = typer.Option(..., help="Harness-Bench task directory name."),
    destination: Path = typer.Option(...),
) -> None:
    source_task = source / "tasks" / task
    if not source_task.is_dir():
        raise typer.BadParameter(f"Task directory not found: {source_task}")

    required = ["task.yaml", "prompt.txt", "fixtures", "oracle_grade.py"]
    missing = [name for name in required if not (source_task / name).exists()]
    if missing:
        raise typer.BadParameter(
            f"Task {task} is missing expected entries: {', '.join(missing)}"
        )
    if destination.exists():
        raise typer.BadParameter(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_task, destination)
    typer.echo(f"Imported {task} to {destination}")


if __name__ == "__main__":
    app()
