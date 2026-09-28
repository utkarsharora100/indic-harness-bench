"""One compatibility contract for the task-grader row in run databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

TASK_GRADE_JOIN = "g.run_id=r.run_id AND g.test_name='task_grader' AND g.kind='task'"
TASK_GRADE_FILTER = "test_name='task_grader' AND kind='task'"


def exact_cell_ids(connection: sqlite3.Connection, experiment_id: str) -> set[str]:
    rows = connection.execute(
        "SELECT cell_id FROM run WHERE experiment_id=?", (experiment_id,)
    ).fetchall()
    return {str(row[0]) for row in rows}


def assert_exact_ids(actual: Iterable[str], expected: Iterable[str], *, label: str) -> None:
    actual_set, expected_set = set(actual), set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} identity mismatch: missing={missing[:5]} extra={extra[:5]} "
            f"(missing_count={len(missing)}, extra_count={len(extra)})"
        )
