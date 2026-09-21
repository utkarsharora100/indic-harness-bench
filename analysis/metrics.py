from __future__ import annotations

import sqlite3
from math import sqrt
from pathlib import Path


FAILURE_CATEGORIES = {
    "instruction_understanding",
    "planning",
    "tool_selection",
    "tool_argument",
    "execution",
    "context_state",
    "error_recovery",
    "completion",
    "evaluation_infrastructure",
}


def validate_category(category: str) -> str:
    if category not in FAILURE_CATEGORIES:
        raise ValueError(f"Unknown failure category: {category}")
    return category


def paired_outcome(english_success: bool, hindi_success: bool, hinglish_success: bool) -> str:
    values = (english_success, hindi_success, hinglish_success)
    if values == (True, True, True):
        return "all_pass"
    if values == (True, False, True):
        return "hindi_only_failure"
    if values == (True, True, False):
        return "hinglish_only_failure"
    if values == (True, False, False):
        return "both_indic_fail"
    if values == (False, True, True):
        return "english_only_failure"
    return "mixed"


def load_runs(database: Path) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute("SELECT * FROM run").fetchall()]
    finally:
        connection.close()


def success_rates(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, int] = {}
    successes: dict[str, int] = {}
    for row in rows:
        # Infrastructure errors are not model outcomes.  Legacy rows do not
        # have a status column, so missing status remains compatible.
        if row.get("status") not in (None, "", "completed", "legacy"):
            continue
        language = row["language"]
        totals[language] = totals.get(language, 0) + 1
        successes[language] = successes.get(language, 0) + int(row["success"])
    return {language: successes[language] / totals[language] for language in totals}


def language_deltas(rows: list[dict]) -> dict[str, float]:
    rates = success_rates(rows)
    english = rates.get("english", 0.0)
    return {
        "hindi_minus_english": rates.get("hindi", 0.0) - english,
        "hinglish_minus_english": rates.get("hinglish", 0.0) - english,
    }


def proportion_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be within [0, total]")

    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)
