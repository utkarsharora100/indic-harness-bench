from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml

from benchmark.translation import missing_entities

app = typer.Typer(no_args_is_help=True)


@app.command()
def import_task(
    source: Path = typer.Option(..., exists=True, file_okay=False),
    task: str = typer.Option(..., help="Harness-Bench task directory name."),
    destination: Path = typer.Option(...),
    translations: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        help="YAML file with english, hindi, and hinglish instruction fields.",
    ),
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
    with translations.open("r", encoding="utf-8") as handle:
        translation_data = yaml.safe_load(handle) or {}
    instruction = translation_data.get("instruction", translation_data)
    required_languages = ("english", "hindi", "hinglish")
    missing_languages = [language for language in required_languages if not instruction.get(language)]
    if missing_languages:
        raise typer.BadParameter(
            f"Translation file is missing: {', '.join(missing_languages)}"
        )

    prompt = (source_task / "prompt.txt").read_text(encoding="utf-8")
    invalid_variants = {
        language: sorted(missing_entities(prompt, instruction[language]))
        for language in required_languages
    }
    invalid_variants = {key: value for key, value in invalid_variants.items() if value}
    if invalid_variants:
        raise typer.BadParameter(
            f"Translations dropped protected technical entities: {invalid_variants}"
        )

    with (source_task / "task.yaml").open("r", encoding="utf-8") as handle:
        upstream_definition = yaml.safe_load(handle) or {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_task, destination / "source")
    definition = {
        "task_id": task,
        "category": "upstream_" + (upstream_definition.get("tags") or ["task"])[0],
        "source_task": f"Qihoo360/harness-bench:{task}",
        "instruction": {language: instruction[language] for language in required_languages},
        "environment": {
            "image": "python:3.12-slim",
            "workspace": "source/fixtures",
            "fixtures": "source/fixtures",
        },
        "limits": {
            "max_steps": 20,
            "timeout_seconds": int(upstream_definition.get("timeout_sec", 600)),
        },
        "upstream": {
            "source_dir": "source",
            "prompt_file": str(upstream_definition.get("prompt_file", "prompt.txt")),
            "fixtures_dir": str(upstream_definition.get("fixtures_dir", "fixtures")),
            "oracle_module": str(upstream_definition.get("oracle_module", "oracle_grade.py")),
            "expected_outcome_score": 1.0,
        },
    }
    with (destination / "task.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(definition, handle, allow_unicode=True, sort_keys=False)
    typer.echo(f"Imported {task} to {destination} with Indic instruction overlays")


if __name__ == "__main__":
    app()
