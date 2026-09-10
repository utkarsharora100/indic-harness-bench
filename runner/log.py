from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
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
    success INTEGER NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    tool_calls INTEGER NOT NULL,
    failed_tool_calls INTEGER NOT NULL,
    execution_time REAL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
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
    details TEXT,
    PRIMARY KEY (run_id, test_name)
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
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def add_task(self, task_id: str, category: str, source_task: str | None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO task(task_id, category, source_task) VALUES (?, ?, ?)",
            (task_id, category, source_task),
        )

    def add_language_variant(self, task_id: str, language: str, instruction: str) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO language_variant(task_id, language, instruction)
               VALUES (?, ?, ?)""",
            (task_id, language, instruction),
        )

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
                success, tool_calls, failed_tool_calls, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?)""",
            (run_id, task_id, language, model, agent, seed, start_time, "{}"),
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
    ) -> None:
        self.connection.execute(
            """INSERT INTO event(
                run_id, step, event_type, tool, arguments_json, result_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                step,
                event_type,
                tool,
                json.dumps(arguments, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                timestamp,
            ),
        )

    def add_grade(self, run_id: str, test_name: str, passed: bool, details: str) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO grade(run_id, test_name, passed, details)
               VALUES (?, ?, ?, ?)""",
            (run_id, test_name, int(passed), details),
        )

    def finish_run(self, run_id: str, **values: Any) -> None:
        self.connection.execute(
            """UPDATE run SET
                end_time=?, success=?, input_tokens=?, output_tokens=?, total_tokens=?,
                tool_calls=?, failed_tool_calls=?, execution_time=?, metadata_json=?
               WHERE run_id=?""",
            (
                values["end_time"], int(values["success"]), values.get("input_tokens"),
                values.get("output_tokens"), values.get("total_tokens"),
                values["tool_calls"], values["failed_tool_calls"], values["execution_time"],
                json.dumps(values.get("metadata", {}), ensure_ascii=False), run_id,
            ),
        )

    def commit(self) -> None:
        self.connection.commit()
