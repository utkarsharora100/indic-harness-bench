from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment (
    experiment_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS task (
    task_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    source_task TEXT
);
CREATE TABLE IF NOT EXISTS language_variant (
    task_id TEXT NOT NULL,
    language TEXT NOT NULL,
    instruction TEXT NOT NULL,
    review_status TEXT,
    reviewer TEXT,
    PRIMARY KEY (task_id, language)
);
CREATE TABLE IF NOT EXISTS run (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    language TEXT NOT NULL,
    model TEXT NOT NULL,
    agent TEXT NOT NULL,
    seed INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    -- NULL means that success is unknown because the cell ended as an
    -- infrastructure error. Model/task failure is represented by 0 only for
    -- completed, gradable cells.
    success INTEGER DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    initial_prompt_tokens INTEGER,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    failed_tool_calls INTEGER NOT NULL DEFAULT 0,
    execution_time REAL,
    agent_time REAL,
    end_to_end_time REAL,
    workspace_sha256 TEXT,
    trace_path TEXT,
    model_metadata_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    experiment_id TEXT,
    cell_id TEXT,
    repetition INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'legacy'
);
CREATE TABLE IF NOT EXISTS attempt (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    error_type TEXT,
    error_message TEXT,
    agent_time REAL,
    end_to_end_time REAL
);
CREATE TABLE IF NOT EXISTS event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    attempt_id TEXT,
    step INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    tool TEXT,
    arguments_json TEXT,
    result_json TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grade (
    run_id TEXT NOT NULL,
    test_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score REAL,
    kind TEXT NOT NULL DEFAULT 'task',
    status TEXT NOT NULL DEFAULT 'completed',
    details TEXT,
    PRIMARY KEY (run_id, test_name)
);
CREATE TABLE IF NOT EXISTS process_grade (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    judge_model TEXT,
    rubric_sha256 TEXT,
    input_sha256 TEXT,
    tool_use_appropriate REAL,
    consistency REAL,
    robustness REAL,
    security_score REAL,
    process_score REAL,
    combined_score REAL,
    details TEXT
);
CREATE TABLE IF NOT EXISTS failure (
    run_id TEXT NOT NULL,
    failure_category TEXT NOT NULL,
    failure_subcategory TEXT,
    annotator TEXT,
    notes TEXT,
    PRIMARY KEY (run_id, failure_category, failure_subcategory)
);
"""


class RunStore:
    """SQLite persistence for stable experiment cells and their attempts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self._migrate_legacy_schema()

    def _migrate_legacy_schema(self) -> None:
        additions = {
            "run": {
                "initial_prompt_tokens": "INTEGER",
                "agent_time": "REAL",
                "end_to_end_time": "REAL",
                "workspace_sha256": "TEXT",
                "trace_path": "TEXT",
                "model_metadata_json": "TEXT",
                "experiment_id": "TEXT",
                "cell_id": "TEXT",
                "repetition": "INTEGER",
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "status": "TEXT NOT NULL DEFAULT 'legacy'",
            },
            "event": {"attempt_id": "TEXT"},
            "grade": {
                "score": "REAL",
                "kind": "TEXT NOT NULL DEFAULT 'task'",
                "status": "TEXT NOT NULL DEFAULT 'completed'",
            },
        }
        for table, columns in additions.items():
            existing = {
                row[1] for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, declaration in columns.items():
                if column not in existing:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                    )
        self._make_success_nullable()
        self.connection.commit()

    def _make_success_nullable(self) -> None:
        """Migrate prototype databases so infrastructure outcomes stay unknown."""
        columns = self.connection.execute("PRAGMA table_info(run)").fetchall()
        success = next((row for row in columns if row[1] == "success"), None)
        if success is None or int(success[3]) == 0:
            return
        names = [row[1] for row in columns]
        self.connection.execute("ALTER TABLE run RENAME TO run_nonnull_legacy")
        self.connection.execute(
            """CREATE TABLE run (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                language TEXT NOT NULL,
                model TEXT NOT NULL,
                agent TEXT NOT NULL,
                seed INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                success INTEGER DEFAULT 0,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                initial_prompt_tokens INTEGER,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                failed_tool_calls INTEGER NOT NULL DEFAULT 0,
                execution_time REAL,
                agent_time REAL,
                end_to_end_time REAL,
                workspace_sha256 TEXT,
                trace_path TEXT,
                model_metadata_json TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                experiment_id TEXT,
                cell_id TEXT,
                repetition INTEGER,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'legacy'
            )"""
        )
        current = [
            "run_id", "task_id", "language", "model", "agent", "seed", "start_time",
            "end_time", "success", "input_tokens", "output_tokens", "total_tokens",
            "initial_prompt_tokens", "tool_calls", "failed_tool_calls", "execution_time",
            "agent_time", "end_to_end_time", "workspace_sha256", "trace_path",
            "model_metadata_json", "metadata_json", "experiment_id", "cell_id",
            "repetition", "attempt_count", "status",
        ]
        preserved = [name for name in current if name in names]
        fields = ", ".join(preserved)
        self.connection.execute(f"INSERT INTO run({fields}) SELECT {fields} FROM run_nonnull_legacy")
        self.connection.execute("DROP TABLE run_nonnull_legacy")

    def close(self) -> None:
        self.connection.close()

    def add_task(self, task_id: str, category: str, source_task: str | None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO task(task_id, category, source_task) VALUES (?, ?, ?)",
            (task_id, category, source_task),
        )

    def add_language_variant(
        self,
        task_id: str,
        language: str,
        instruction: str,
        review_status: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO language_variant(
                task_id, language, instruction, review_status, reviewer
            ) VALUES (?, ?, ?, ?, ?)""",
            (task_id, language, instruction, review_status, reviewer),
        )

    def ensure_experiment(
        self,
        experiment_id: str,
        config: dict[str, Any],
        manifest: dict[str, Any],
        created_at: str,
    ) -> None:
        existing = self.connection.execute(
            "SELECT config_json, manifest_json FROM experiment WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
        manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        if existing is not None and (
            existing["config_json"] != config_json or existing["manifest_json"] != manifest_json
        ):
            raise ValueError(
                f"Experiment {experiment_id!r} already exists with different configuration or manifest"
            )
        self.connection.execute(
            """INSERT OR IGNORE INTO experiment(
                experiment_id, config_json, manifest_json, created_at, status
            ) VALUES (?, ?, ?, ?, 'active')""",
            (experiment_id, config_json, manifest_json, created_at),
        )

    def ensure_cell(
        self,
        *,
        cell_id: str,
        experiment_id: str,
        task_id: str,
        language: str,
        model: str,
        agent: str,
        repetition: int,
        seed: int,
    ) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO run(
                run_id, task_id, language, model, agent, seed, start_time,
                success, tool_calls, failed_tool_calls, metadata_json,
                experiment_id, cell_id, repetition, status
            ) VALUES (?, ?, ?, ?, ?, ?, '', 0, 0, 0, '{}', ?, ?, ?, 'pending')""",
            (cell_id, task_id, language, model, agent, seed, experiment_id, cell_id, repetition),
        )

    def get_cell(self, cell_id: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM run WHERE run_id = ?", (cell_id,)).fetchone()

    def start_cell(self, cell_id: str, start_time: str, attempt_count: int) -> None:
        self.connection.execute(
            """UPDATE run SET status='running', start_time=?, attempt_count=? WHERE run_id=?""",
            (start_time, attempt_count, cell_id),
        )

    def start_attempt(
        self,
        attempt_id: str,
        run_id: str,
        attempt_no: int,
        start_time: str,
    ) -> None:
        self.connection.execute(
            """INSERT INTO attempt(
                attempt_id, run_id, attempt_no, status, start_time
            ) VALUES (?, ?, ?, 'running', ?)""",
            (attempt_id, run_id, attempt_no, start_time),
        )

    def recover_interrupted(self, experiment_id: str, end_time: str) -> int:
        """Close rows left running when an orchestration process was interrupted."""
        run_filter = "SELECT run_id FROM run WHERE experiment_id = ?"
        attempts = self.connection.execute(
            f"""UPDATE attempt SET status='infrastructure_error', end_time=?,
                error_type='InterruptedRun', error_message=?
                WHERE status='running' AND run_id IN ({run_filter})""",
            (end_time, "orchestration process ended before the attempt completed", experiment_id),
        ).rowcount
        self.connection.execute(
            f"""UPDATE run SET status='pending' WHERE status='running'
                AND experiment_id = ?""",
            (experiment_id,),
        )
        return int(attempts or 0)

    def finish_attempt(self, attempt_id: str, **values: Any) -> None:
        self.connection.execute(
            """UPDATE attempt SET status=?, end_time=?, error_type=?, error_message=?,
                agent_time=?, end_to_end_time=? WHERE attempt_id=?""",
            (
                values["status"],
                values.get("end_time"),
                values.get("error_type"),
                values.get("error_message"),
                values.get("agent_time"),
                values.get("end_to_end_time"),
                attempt_id,
            ),
        )

    def finish_cell(self, cell_id: str, **values: Any) -> None:
        success = values.get("success")
        self.connection.execute(
            """UPDATE run SET end_time=?, status=?, success=?, input_tokens=?, output_tokens=?,
                total_tokens=?, initial_prompt_tokens=?, tool_calls=?, failed_tool_calls=?,
                execution_time=?, agent_time=?, end_to_end_time=?, workspace_sha256=?,
                trace_path=?, model_metadata_json=?, metadata_json=? WHERE run_id=?""",
            (
                values.get("end_time"),
                values["status"],
                None if success is None else int(success),
                values.get("input_tokens"),
                values.get("output_tokens"),
                values.get("total_tokens"),
                values.get("initial_prompt_tokens"),
                values.get("tool_calls", 0),
                values.get("failed_tool_calls", 0),
                values.get("execution_time"),
                values.get("agent_time"),
                values.get("end_to_end_time"),
                values.get("workspace_sha256"),
                values.get("trace_path"),
                json.dumps(values.get("model_metadata", {}), ensure_ascii=False),
                json.dumps(values.get("metadata", {}), ensure_ascii=False),
                cell_id,
            ),
        )

    def add_event(
        self,
        run_id: str,
        step: int,
        event_type: str,
        timestamp: str,
        tool: str | None = None,
        arguments: Any = None,
        result: Any = None,
        attempt_id: str | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO event(
                run_id, attempt_id, step, event_type, tool, arguments_json, result_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                attempt_id,
                step,
                event_type,
                tool,
                json.dumps(arguments, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                timestamp,
            ),
        )

    def add_grade(
        self,
        run_id: str,
        test_name: str,
        passed: bool,
        details: str,
        *,
        score: float | None = None,
        kind: str = "task",
        status: str = "completed",
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO grade(
                   run_id, test_name, passed, score, kind, status, details
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, test_name, int(passed), score, kind, status, details),
        )

    def add_process_grade(self, run_id: str, values: dict[str, Any]) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO process_grade(
                run_id, status, judge_model, rubric_sha256, input_sha256,
                tool_use_appropriate, consistency, robustness, security_score,
                process_score, combined_score, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                values.get("status", "missing"),
                values.get("judge_model"),
                values.get("rubric_sha256"),
                values.get("input_sha256"),
                values.get("tool_use_appropriate"),
                values.get("consistency"),
                values.get("robustness"),
                values.get("security_score"),
                values.get("process_score"),
                values.get("combined_score"),
                json.dumps(values, ensure_ascii=False, sort_keys=True),
            ),
        )

    # Compatibility methods for the original prototype and old pilot data.
    def start_run(
        self,
        run_id: str,
        task_id: str,
        language: str,
        model: str,
        agent: str,
        seed: int,
        start_time: str,
    ) -> None:
        self.connection.execute(
            """INSERT INTO run(
                run_id, task_id, language, model, agent, seed, start_time,
                success, tool_calls, failed_tool_calls, metadata_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, '{}', 'running')""",
            (run_id, task_id, language, model, agent, seed, start_time),
        )

    def finish_run(self, run_id: str, **values: Any) -> None:
        self.connection.execute(
            """UPDATE run SET
                end_time=?, success=?, input_tokens=?, output_tokens=?, total_tokens=?,
                tool_calls=?, failed_tool_calls=?, execution_time=?, metadata_json=?, status=?
               WHERE run_id=?""",
            (
                values["end_time"],
                int(values["success"]),
                values.get("input_tokens"),
                values.get("output_tokens"),
                values.get("total_tokens"),
                values["tool_calls"],
                values["failed_tool_calls"],
                values["execution_time"],
                json.dumps(values.get("metadata", {}), ensure_ascii=False),
                "completed",
                run_id,
            ),
        )

    def rows(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id is None:
            query = "SELECT * FROM run ORDER BY start_time, run_id"
            params: tuple[Any, ...] = ()
        else:
            query = "SELECT * FROM run WHERE experiment_id = ? ORDER BY start_time, run_id"
            params = (experiment_id,)
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def commit(self) -> None:
        self.connection.commit()
