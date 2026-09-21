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


def load_corrected_rows(database: Path, experiment_id: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        query = """
            SELECT r.*, g.score AS outcome_score, g.status AS grade_status,
                   pg.status AS process_status, pg.process_score, pg.security_score,
                   pg.combined_score
            FROM run r
            JOIN grade g ON g.run_id = r.run_id AND g.kind = 'task'
            LEFT JOIN process_grade pg ON pg.run_id = r.run_id
            WHERE r.experiment_id = ? AND r.status = 'completed'
        """
        return [dict(row) for row in connection.execute(query, (experiment_id,))]
    finally:
        connection.close()


def task_balanced_outcome(rows: list[dict[str, Any]]) -> dict[str, float]:
    return task_balanced_metric(rows, "outcome_score", "grade_status")


def task_balanced_metric(
    rows: list[dict[str, Any]],
    metric: str,
    status_field: str | None = None,
) -> dict[str, float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        if status_field is not None and row.get(status_field) not in (None, "completed"):
            continue
        grouped[(row["task_id"], row["language"], row["agent"])].append(float(value))
    result: dict[str, float] = {}
    for agent in sorted({key[2] for key in grouped}):
        for language in LANGUAGES:
            values = [
                mean(scores)
                for (task, lang, harness), scores in grouped.items()
                if harness == agent and lang == language
            ]
            if values:
                result[f"{agent}:{language}"] = mean(values)
    return result


def task_condition_scores(
    rows: list[dict[str, Any]],
    metric: str = "outcome_score",
    status_field: str | None = "grade_status",
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        if status_field is not None and row.get(status_field) not in (None, "completed"):
            continue
        grouped[(row["task_id"], row["agent"], row["language"])].append(float(value))
    return {key: mean(values) for key, values in grouped.items()}


def paired_outcome_deltas(rows: list[dict[str, Any]], *, metric: str = "outcome_score") -> dict[str, dict[str, float]]:
    status_field = "grade_status" if metric == "outcome_score" else "process_status"
    grouped = task_condition_scores(rows, metric, status_field)
    by_harness: dict[str, dict[str, float]] = {}
    for agent in sorted({key[1] for key in grouped}):
        task_values: dict[str, dict[str, float]] = defaultdict(dict)
        for (task, harness, language), value in grouped.items():
            if harness == agent:
                task_values[task][language] = value
        english = [values["english"] for values in task_values.values() if "english" in values]
        deltas: dict[str, float] = {}
        for language in ("hindi", "hinglish"):
            paired = [values[language] - values["english"] for values in task_values.values() if language in values and "english" in values]
            if paired:
                deltas[language] = mean(paired)
        if english:
            by_harness[agent] = deltas
    return by_harness


def score_difference_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = task_condition_scores(rows)
    result: list[dict[str, Any]] = []
    task_harnesses = sorted({(task, harness) for task, harness, _language in scores})
    for task, harness in task_harnesses:
        english = scores.get((task, harness, "english"))
        if english is None:
            continue
        for language in ("hindi", "hinglish"):
            value = scores.get((task, harness, language))
            if value is not None:
                result.append(
                    {
                        "task_id": task,
                        "agent": harness,
                        "language": language,
                        "difference": value - english,
                    }
                )
    return result


def harness_interactions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = task_condition_scores(rows)
    harnesses = sorted({key[1] for key in scores})
    means = task_balanced_metric(rows, "outcome_score", "grade_status")
    within_language: dict[str, float] = {}
    for harness in harnesses:
        for language in LANGUAGES:
            key = f"{harness}:{language}"
            baseline = means.get(f"react:{language}")
            if key in means and baseline is not None:
                within_language[key] = means[key] - baseline
    language_by_harness: dict[str, float] = {}
    react_delta = paired_outcome_deltas(rows).get("react", {})
    for harness in harnesses:
        if harness == "react":
            continue
        deltas = paired_outcome_deltas(rows).get(harness, {})
        for language in ("hindi", "hinglish"):
            if language in deltas and language in react_delta:
                language_by_harness[f"{harness}:{language}"] = deltas[language] - react_delta[language]
    return {
        "task_balanced_harness_minus_react": within_language,
        "language_by_harness_difference_in_differences": language_by_harness,
    }


def trace_links(database: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_root = database.parent / "results"
    return [
        {
            "cell_id": row.get("cell_id") or row["run_id"],
            "task_id": row["task_id"],
            "language": row["language"],
            "agent": row["agent"],
            "repetition": row.get("repetition"),
            "trace": row.get("trace_path"),
            "result": str(result_root / f"{row.get('cell_id') or row['run_id']}.json"),
        }
        for row in rows
    ]


def usage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    fields = (
        "initial_prompt_tokens",
        "total_tokens",
        "agent_time",
        "end_to_end_time",
        "tool_calls",
        "failed_tool_calls",
    )
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is not None:
                grouped[(row["agent"], row["language"])][field].append(float(value))
    return {
        f"{agent}:{language}": {
            field: (mean(values) if values else None)
            for field, values in grouped[(agent, language)].items()
        }
        for agent, language in sorted(grouped)
    }


def bootstrap_paired_deltas(
    rows: list[dict[str, Any]],
    *,
    metric: str = "outcome_score",
    repetitions: int = 20_000,
    seed: int = 1701,
) -> dict[str, dict[str, dict[str, float]]]:
    """Task-cluster bootstrap of paired language deltas.

    The same sampled task indices are used for English, Hindi, and Hinglish;
    independent marginal resampling would manufacture uncertainty in paired
    contrasts when the outcomes are identical.
    """
    status_field = "grade_status" if metric == "outcome_score" else "process_status"
    scores = task_condition_scores(rows, metric, status_field)
    result: dict[str, dict[str, dict[str, float]]] = {}
    rng = random.Random(seed)
    for agent in sorted({key[1] for key in scores}):
        tasks: dict[str, dict[str, float]] = defaultdict(dict)
        for (task, harness, language), value in scores.items():
            if harness == agent:
                tasks[task][language] = value
        eligible = [task for task, values in tasks.items() if all(lang in values for lang in LANGUAGES)]
        if not eligible:
            continue
        samples = {language: [] for language in ("hindi", "hinglish")}
        for _ in range(repetitions):
            chosen = [rng.choice(eligible) for _ in eligible]
            for language in samples:
                samples[language].append(mean(tasks[task][language] - tasks[task]["english"] for task in chosen))
        result[agent] = {}
        for language, values in samples.items():
            values.sort()
            result[agent][language] = {
                "lower": values[int(0.025 * (len(values) - 1))],
                "upper": values[int(0.975 * (len(values) - 1))],
            }
    return result


def paired_outcome_categories(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    cells: dict[tuple[str, str, int | None], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("grade_status") not in (None, "completed"):
            continue
        cells[(row["task_id"], row["agent"], row.get("repetition"))][row["language"]] = bool(row["success"])
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_task, agent, _rep), values in cells.items():
        if all(language in values for language in LANGUAGES):
            result[agent][paired_outcome(values["english"], values["hindi"], values["hinglish"])] += 1
    return {agent: dict(sorted(values.items())) for agent, values in sorted(result.items())}


def error_conditioned_recovery(database: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        grouped: dict[str, list[bool]] = defaultdict(list)

        def result_payload(raw: str | None) -> dict[str, Any] | None:
            try:
                value = json.loads(raw or "null")
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None

        for row in rows:
            events = [dict(event) for event in connection.execute(
                "SELECT event_type, result_json FROM event WHERE run_id = ? ORDER BY event_id",
                (row["run_id"],),
            )]
            observations: list[dict[str, Any]] = []
            for event in events:
                if event["event_type"] == "tool_call":
                    payload = result_payload(event["result_json"])
                    if payload is not None:
                        observations.append(payload)
                elif event["event_type"] == "proxy_request":
                    payload = result_payload(event["result_json"])
                    for message in (payload or {}).get("messages", []):
                        if not isinstance(message, dict) or message.get("role") != "tool":
                            continue
                        tool_result = result_payload(message.get("content"))
                        if tool_result is not None:
                            observations.append(tool_result)
            failed = [index for index, payload in enumerate(observations) if payload.get("ok") is False]
            if not failed:
                continue
            recovered = any(
                any(
                    later.get("ok") is True
                    for later in observations[index + 1 :]
                )
                for index in failed
            )
            grouped[row["agent"]].append(recovered)
        return {
            agent: {"error_runs": len(values), "recovered_runs": sum(values), "rate": mean(values) if values else None}
            for agent, values in sorted(grouped.items())
        }
    finally:
        connection.close()
