from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from analysis.metrics import language_deltas, load_runs, success_rates
from analysis.phase1 import (
    bootstrap_task_intervals,
    load_gradable_rows,
    paired_language_deltas,
    paired_outcomes,
    secondary_metrics,
    task_balanced_success,
)


def build_report(database: Path, output: Path) -> None:
    """Keep the original prototype report command working for legacy data."""
    rows = load_runs(database)
    output.parent.mkdir(parents=True, exist_ok=True)
    rates = success_rates(rows)
    lines = [
        "# Phase 1 Results",
        "",
        "| Language | Success rate |",
        "|---|---:|",
    ]
    for language, rate in sorted(rates.items()):
        lines.append(f"| {language} | {rate:.4f} |")
    lines.extend(["", "## Language deltas", "", "```text", repr(language_deltas(rows)), "```"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _experiment_id(database: Path) -> str | None:
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT experiment_id FROM run WHERE experiment_id IS NOT NULL LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        connection.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_phase1_report(database: Path, output: Path) -> dict[str, Any]:
    experiment_id = _experiment_id(database)
    rows = load_gradable_rows(database, experiment_id)
    connection = sqlite3.connect(database)
    try:
        total_cells = connection.execute(
            "SELECT COUNT(*) FROM run WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()[0]
        status_counts = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT status, COUNT(*) FROM run WHERE experiment_id = ? GROUP BY status",
                (experiment_id,),
            ).fetchall()
        }
    finally:
        connection.close()

    report: dict[str, Any] = {
        "provisional": True,
        "translation_review_status": "unreviewed",
        "experiment_id": experiment_id,
        "cells": {
            "database_rows": total_cells,
            "gradable_completed": len(rows),
            "statuses": status_counts,
        },
        "cell_success_counts": {
            language: {
                "successes": sum(
                    1 for row in rows if row["language"] == language and bool(row["success"])
                ),
                "cells": sum(1 for row in rows if row["language"] == language),
            }
            for language in ("english", "hindi", "hinglish")
        },
        "task_balanced_success": task_balanced_success(rows),
        "language_deltas": paired_language_deltas(rows),
        "bootstrap_95_task_cluster": bootstrap_task_intervals(rows),
        "paired_outcomes": paired_outcomes(rows),
        "secondary_metrics": secondary_metrics(database, rows),
        "trace_directory": str(output.parent / "traces"),
        "notes": [
            "Hindi and Hinglish overlays are provisional and unreviewed.",
            "Infrastructure-error cells are excluded from model success rates.",
            "Failure-stage attribution is reserved for reviewed diagnostics.",
        ],
    }
    report = _json_safe(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rates = report["task_balanced_success"]
    intervals = report["bootstrap_95_task_cluster"]
    deltas = report["language_deltas"]
    lines = [
        "# Phase I provisional report",
        "",
        "> Translation review is outstanding. Treat these findings as provisional; infrastructure errors are not model failures.",
        "",
        f"Experiment: `{experiment_id or 'unknown'}`  ",
        f"Gradable completed cells: **{len(rows)} / {total_cells}**",
        "",
        "## Task-balanced success",
        "",
        "| Language | Successful cells | Gradable cells | Task-balanced rate | 95% task-cluster interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for language in ("english", "hindi", "hinglish"):
        rate = rates.get(language)
        interval = intervals.get(language)
        counts = report["cell_success_counts"].get(language, {})
        rate_text = "missing" if rate is None else f"{rate:.3f}"
        interval_text = "missing" if interval is None else f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        lines.append(
            f"| {language} | {counts.get('successes', 'missing')} | "
            f"{counts.get('cells', 'missing')} | {rate_text} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas against English",
            "",
            "| Contrast | Delta |",
            "|---|---:|",
            f"| Hindi − English | {deltas.get('hindi_minus_english', 'missing')} |",
            f"| Hinglish − English | {deltas.get('hinglish_minus_english', 'missing')} |",
            "",
            "## Paired outcomes",
            "",
            "| Outcome | Cells |",
            "|---|---:|",
        ]
    )
    for outcome, count in report["paired_outcomes"].items():
        lines.append(f"| {outcome} | {count} |")
    secondary = report["secondary_metrics"]
    secondary_rows = (
        ("Initial-prompt tokens", "initial_prompt_tokens"),
        ("Total tokens", "total_tokens"),
        ("Agent time (s)", "agent_time_seconds"),
        ("End-to-end time (s)", "end_to_end_time_seconds"),
        ("Tool calls", "tool_calls"),
        ("Failed tool calls", "failed_tool_calls"),
        ("Recovery after error", "recovery_after_error"),
    )
    lines.extend(
        [
            "",
            "## Secondary metrics by language",
            "",
            "| Metric | English | Hindi | Hinglish |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in secondary_rows:
        values = secondary.get(key, {})
        formatted = []
        for language in ("english", "hindi", "hinglish"):
            value = values.get(language)
            if value is None:
                formatted.append("missing")
            elif key == "recovery_after_error":
                formatted.append(f"{float(value):.3f}")
            else:
                formatted.append(f"{float(value):.3f}")
        lines.append(f"| {label} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    lines.extend(
        [
            "",
            "The JSON companion contains the same values with machine-readable precision. Missing usage remains `null`/missing rather than zero.",
            "",
            "## Provenance and traces",
            "",
            "Each completed cell has a stable `cell_id`; its raw JSONL trace is stored beside this report under the ignored experiment storage directory. Reviewers should annotate observed failure stages before making causal claims.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return report
