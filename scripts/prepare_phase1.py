from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from benchmark.models import TaskDefinition
from benchmark.translation import validate_translation


REQUIRED_SOURCE_FILES = ("task.yaml", "prompt.txt", "fixtures", "oracle_grade.py")
LANGUAGES = ("english", "hindi", "hinglish")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_revision(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Source is not a usable Git checkout: {source}\n{result.stderr}")
    return result.stdout.strip()


def load_selection(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = data.get("source") or {}
    tasks = data.get("tasks") or []
    if not source.get("revision"):
        raise ValueError("Selection manifest must pin source.revision")
    if len(tasks) != int(data.get("target_count", 0)):
        raise ValueError("Selection manifest target_count does not match task count")
    if len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("Selection manifest contains duplicate task IDs")
    return data


def load_translations(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise ValueError("Translation manifest tasks must be a mapping")
    return data


def validate_python_asset(path: Path, required_callable: str | None = None) -> None:
    module_name = f"phase1_asset_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import pinned Python asset: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"Pinned Python asset failed to import: {path}: {exc}") from exc
    if required_callable and not callable(getattr(module, required_callable, None)):
        raise ValueError(f"Pinned Python asset lacks {required_callable}(): {path}")


def validate_categories(selection: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for item in selection["tasks"]:
        category = item["category"]
        counts[category] = counts.get(category, 0) + 1
    for category, expected in (selection.get("quotas") or {}).items():
        if counts.get(category, 0) != int(expected):
            raise ValueError(
                f"Category quota mismatch for {category}: got {counts.get(category, 0)}, expected {expected}"
            )


def build_task_definition(
    task_id: str,
    category: str,
    revision: str,
    source_definition: dict[str, Any],
    instructions: dict[str, str],
) -> dict[str, Any]:
    timeout = int(source_definition.get("timeout_sec", 600))
    prompt_file = str(source_definition.get("prompt_file", "prompt.txt"))
    prompt_files = [str(value) for value in (source_definition.get("prompt_files") or [])]
    fixtures_dir = str(source_definition.get("fixtures_dir", "fixtures"))
    oracle_module = str(source_definition.get("oracle_module", "oracle_grade.py"))
    hooks_module = str(source_definition.get("hooks_module", "hooks.py"))
    return {
        "task_id": task_id,
        "category": category,
        "source_task": f"Qihoo360/harness-bench@{revision}:{task_id}",
        "instruction": instructions,
        "environment": {
            "image": "indic-harness-phase1:2026-09-21",
            "workspace": f"source/{fixtures_dir}",
            "fixtures": f"source/{fixtures_dir}",
        },
        "limits": {"max_steps": 40, "timeout_seconds": timeout},
        "upstream": {
            "source_dir": "source",
            "prompt_file": prompt_file,
            "prompt_files": prompt_files,
            "fixtures_dir": fixtures_dir,
            "oracle_module": oracle_module,
            "hooks_module": hooks_module,
            "expected_outcome_score": 1.0,
        },
    }


def prepare(
    source: Path,
    selection_path: Path,
    translations_path: Path,
    destination: Path,
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    selection = load_selection(selection_path)
    validate_categories(selection)
    translations = load_translations(translations_path)
    revision = source_revision(source)
    expected_revision = selection["source"]["revision"]
    if revision != expected_revision:
        raise RuntimeError(
            f"Source revision mismatch: checkout is {revision}, manifest requires {expected_revision}"
        )

    prepared: list[dict[str, Any]] = []
    upstream_default_rubric = source / "grading" / "default_rubric.py"
    if not upstream_default_rubric.is_file():
        raise RuntimeError(f"Pinned source is missing grading/default_rubric.py: {upstream_default_rubric}")
    if not check_only:
        rubric_destination = destination / "grading" / "default_rubric.py"
        rubric_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream_default_rubric, rubric_destination)
    for item in selection["tasks"]:
        task_id = item["task_id"]
        source_task = source / "tasks" / task_id
        if not source_task.is_dir():
            raise FileNotFoundError(f"Selected task does not exist: {source_task}")
        missing = [name for name in REQUIRED_SOURCE_FILES if not (source_task / name).exists()]
        if missing:
            raise ValueError(f"{task_id} is missing required files: {', '.join(missing)}")
        source_definition = yaml.safe_load(
            (source_task / "task.yaml").read_text(encoding="utf-8")
        ) or {}
        validate_python_asset(source_task / str(source_definition.get("oracle_module", "oracle_grade.py")), "score_workspace")
        hooks_path = source_task / str(source_definition.get("hooks_module", "hooks.py"))
        if hooks_path.is_file():
            validate_python_asset(hooks_path)
        rubric_path = source_task / "llm_rubric.py"
        if rubric_path.is_file():
            validate_python_asset(rubric_path)
        source_hash = tree_sha256(source_task)
        expected_hash = item.get("source_sha256")
        if expected_hash and expected_hash != source_hash:
            raise RuntimeError(
                f"Source hash mismatch for {task_id}: checkout has {source_hash}, manifest has {expected_hash}"
            )

        translation_record = translations.get("tasks", {}).get(task_id)
        if not isinstance(translation_record, dict):
            raise ValueError(f"Missing translation record for {task_id}")
        if translation_record.get("source_sha256") != source_hash:
            raise RuntimeError(f"Translation source hash mismatch for {task_id}")
        instructions = translation_record.get("instructions") or {}
        if any(not isinstance(instructions.get(language), str) for language in LANGUAGES):
            raise ValueError(f"Translation record for {task_id} is missing a language")
        prompt = (source_task / "prompt.txt").read_text(encoding="utf-8")
        if instructions["english"] != prompt:
            raise ValueError(f"English translation changed the canonical prompt for {task_id}")
        for language in ("hindi", "hinglish"):
            errors = validate_translation(prompt, instructions[language])
            if errors:
                raise ValueError(f"Invalid {language} translation for {task_id}: {'; '.join(errors)}")

        definition = build_task_definition(
            task_id, item["category"], revision, source_definition, instructions
        )
        target = destination / task_id
        if not check_only:
            if target.exists():
                existing_hash_path = target / ".source_sha256"
                if not existing_hash_path.is_file() or existing_hash_path.read_text().strip() != source_hash:
                    raise RuntimeError(f"Prepared task already exists with a different source: {target}")
            else:
                target.mkdir(parents=True, exist_ok=False)
                shutil.copytree(source_task, target / "source")
                (target / ".source_sha256").write_text(source_hash + "\n", encoding="utf-8")
            # Refresh the generated wrapper even when the ignored cache was
            # prepared by an older runner version.  The pinned source tree is
            # still protected by the hash check above.
            (target / "task.yaml").write_text(
                yaml.safe_dump(definition, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            TaskDefinition.from_file(target / "task.yaml")
        prepared.append({"task_id": task_id, "category": item["category"], "source_sha256": source_hash})

    result = {
        "schema_version": 1,
        "source": {"repository": selection["source"]["repository"], "revision": revision},
        "task_count": len(prepared),
        "tasks": prepared,
    }
    if not check_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / "dataset-manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned Phase I task cache")
    parser.add_argument("--source", type=Path, required=True, help="Pinned Harness-Bench checkout")
    parser.add_argument("--selection", type=Path, default=Path("benchmark/task_selection.yaml"))
    parser.add_argument("--translations", type=Path, default=Path("benchmark/translations/phase1.yaml"))
    parser.add_argument("--destination", type=Path, default=Path("data/phase1/tasks"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source.resolve(), args.selection, args.translations, args.destination, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
