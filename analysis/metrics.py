from __future__ import annotations

import sqlite3
from pathlib import Path


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
