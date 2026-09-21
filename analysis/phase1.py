from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.metrics import paired_outcome


LANGUAGES = ("english", "hindi", "hinglish")


def load_gradable_rows(database: Path, experiment_id: str | None = None) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        clauses = ["status = 'completed'", "EXISTS (SELECT 1 FROM grade g WHERE g.run_id = run.run_id)"]
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        query = "SELECT * FROM run WHERE " + " AND ".join(clauses)
        return [dict(row) for row in connection.execute(query, params).fetchall()]
    finally:
        connection.close()


def task_balanced_success(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_task_language: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_task_language[(row["task_id"], row["language"])].append(float(row["success"]))
    task_ids = sorted({row["task_id"] for row in rows})
    result: dict[str, float] = {}
    for language in LANGUAGES:
        task_values = [
            mean(by_task_language[(task_id, language)])
            for task_id in task_ids
            if by_task_language.get((task_id, language))
        ]
        if task_values:
            result[language] = mean(task_values)
    return result


def paired_language_deltas(rows: list[dict[str, Any]]) -> dict[str, float]:
    rates = task_balanced_success(rows)
    english = rates.get("english")
    if english is None:
        return {}
    return {
        "hindi_minus_english": rates.get("hindi", float("nan")) - english,
        "hinglish_minus_english": rates.get("hinglish", float("nan")) - english,
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def bootstrap_task_intervals(
    rows: list[dict[str, Any]], *, repetitions: int = 2000, seed: int = 17
) -> dict[str, dict[str, float]]:
    by_task_language: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_task_language[(row["task_id"], row["language"])].append(float(row["success"]))
    task_ids = sorted({row["task_id"] for row in rows})
    result: dict[str, dict[str, float]] = {}
    rng = random.Random(seed)
    for language in LANGUAGES:
        task_values = {
            task_id: mean(by_task_language[(task_id, language)])
            for task_id in task_ids
            if by_task_language.get((task_id, language))
        }
        if not task_values:
            continue
        available = list(task_values)
        samples = []
        for _ in range(repetitions):
            sampled = [task_values[rng.choice(available)] for _ in available]
            samples.append(mean(sampled))
        result[language] = {
            "lower": _percentile(samples, 0.025),
            "upper": _percentile(samples, 0.975),
        }
    return result


def paired_outcomes(rows: list[dict[str, Any]]) -> dict[str, int]:
    cells: dict[tuple[str, int | None], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        cells[(row["task_id"], row.get("repetition"))][row["language"]] = bool(row["success"])
    counts: dict[str, int] = defaultdict(int)
    for values in cells.values():
        if all(language in values for language in LANGUAGES):
            counts[paired_outcome(values["english"], values["hindi"], values["hinglish"])] += 1
    return dict(sorted(counts.items()))


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for language in LANGUAGES:
        values = [float(row[field]) for row in rows if row["language"] == language and row.get(field) is not None]
        result[language] = mean(values) if values else None
    return result


def secondary_metrics(database: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in connection.execute(
            "SELECT run_id, event_type, tool, result_json FROM event ORDER BY event_id"
        ).fetchall():
            item = dict(raw)
            try:
                item["result"] = json.loads(item.pop("result_json") or "null")
            except json.JSONDecodeError:
                item["result"] = None
            events[item["run_id"]].append(item)
    finally:
        connection.close()

    recovery_by_language: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        run_events = events.get(row["run_id"], [])
        failed_indices = [
            index
            for index, event in enumerate(run_events)
            if event["event_type"] == "tool_call"
            and isinstance(event.get("result"), dict)
            and event["result"].get("ok") is False
        ]
        recovered = any(
            any(later.get("event_type") == "tool_call" for later in run_events[index + 1 :])
            for index in failed_indices
        )
        recovery_by_language[row["language"]].append(int(recovered))
    return {
        "initial_prompt_tokens": _numeric_summary(rows, "initial_prompt_tokens"),
        "total_tokens": _numeric_summary(rows, "total_tokens"),
        "agent_time_seconds": _numeric_summary(rows, "agent_time"),
        "end_to_end_time_seconds": _numeric_summary(rows, "end_to_end_time"),
        "tool_calls": _numeric_summary(rows, "tool_calls"),
        "failed_tool_calls": _numeric_summary(rows, "failed_tool_calls"),
        "recovery_after_error": {
            language: (mean(values) if values else None)
            for language, values in sorted(recovery_by_language.items())
        },
    }
