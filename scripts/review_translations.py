from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmark.translation import protected_entities


LANGUAGES = ("english", "hindi", "hinglish")


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("tasks"), dict):
        raise ValueError("translation file must contain a tasks mapping")
    return data


def checklist_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, record in data["tasks"].items():
        instructions = record.get("instructions") or {}
        review = record.get("review") or {}
        source = instructions.get("english", "")
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        for language in LANGUAGES:
            variant = instructions.get(language, "")
            language_review = review.get(language) or {}
            rows.append(
                {
                    "task_id": task_id,
                    "language": language,
                    "status": language_review.get(
                        "status", "canonical" if language == "english" else "unreviewed"
                    ),
                    "reviewer": language_review.get("reviewer") or "",
                    "protected_entity_count": len(protected_entities(source)),
                    "variant_sha256": hashlib.sha256(variant.encode("utf-8")).hexdigest(),
                    "source_prompt_sha256": source_hash,
                    "notes": language_review.get("notes") or "",
                }
            )
    return rows


def export_checklist(source: Path, destination: Path) -> None:
    rows = checklist_rows(load(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mark_review(
    source: Path,
    task_id: str,
    language: str,
    status: str,
    reviewer: str,
    notes: str,
) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of: {', '.join(LANGUAGES)}")
    if status not in {"canonical", "unreviewed", "approved", "needs_revision"}:
        raise ValueError("status must be canonical, unreviewed, approved, or needs_revision")
    data = load(source)
    record = data["tasks"].get(task_id)
    if not isinstance(record, dict):
        raise ValueError(f"unknown task: {task_id}")
    instructions = record.get("instructions") or {}
    if not isinstance(instructions.get(language), str) or not instructions[language].strip():
        raise ValueError(f"missing {language} instruction for {task_id}")
    review = record.setdefault("review", {})
    review[language] = {
        "status": status,
        "reviewer": reviewer or None,
        "notes": notes or None,
    }
    # Only review metadata changes.  The canonical prompt, translations, and
    # source hash remain byte-for-byte values in the parsed dataset, so old
    # run records do not need to be regenerated.
    source.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or update Phase I translation review metadata")
    parser.add_argument("--translations", type=Path, default=Path("benchmark/translations/phase1.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="write a CSV review checklist")
    export.add_argument("--output", type=Path, required=True)

    approve = subparsers.add_parser("mark", help="mark one task/language review status")
    approve.add_argument("--task-id", required=True)
    approve.add_argument("--language", required=True, choices=LANGUAGES)
    approve.add_argument("--status", required=True, choices=("canonical", "unreviewed", "approved", "needs_revision"))
    approve.add_argument("--reviewer", default="")
    approve.add_argument("--notes", default="")

    args = parser.parse_args()
    try:
        if args.command == "export":
            export_checklist(args.translations, args.output)
            print(args.output)
        else:
            mark_review(
                args.translations,
                args.task_id,
                args.language,
                args.status,
                args.reviewer,
                args.notes,
            )
            print(f"updated review metadata for {args.task_id}/{args.language}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
