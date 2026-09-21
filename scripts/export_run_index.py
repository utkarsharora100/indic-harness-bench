from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


FIELDS = (
    "cell_id",
    "task_id",
    "language",
    "repetition",
    "status",
    "success",
    "grade_passed",
    "attempt_count",
    "tool_calls",
    "failed_tool_calls",
    "initial_prompt_tokens",
    "total_tokens",
    "agent_time",
    "end_to_end_time",
    "workspace_sha256",
    "trace_path",
    "result_path",
)


def export(database: Path, output: Path, experiment_id: str | None = None) -> int:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        query = """
            SELECT r.cell_id, r.task_id, r.language, r.repetition, r.status, r.success,
                   g.passed AS grade_passed, r.attempt_count, r.tool_calls,
                   r.failed_tool_calls, r.initial_prompt_tokens, r.total_tokens,
                   r.agent_time, r.end_to_end_time, r.workspace_sha256,
                   r.trace_path
            FROM run r
            LEFT JOIN grade g ON g.run_id = r.run_id AND g.test_name = 'task_grader'
        """
        params: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE r.experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY r.task_id, r.language, r.repetition"
        rows = [dict(row) for row in connection.execute(query, params)]
    finally:
        connection.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            row["cell_id"] = row.get("cell_id") or ""
            row["result_path"] = str(output.parent / "results" / f"{row['cell_id']}.json")
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a secret-free index of Phase I run logs")
    parser.add_argument("--database", type=Path, default=Path("data/phase1/experiment/runs.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/phase1/experiment/run-index.csv"))
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    print(export(args.database, args.output, args.experiment_id))


if __name__ == "__main__":
    main()
