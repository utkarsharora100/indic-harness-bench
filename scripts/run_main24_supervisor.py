"""Durable, resumable Phase I gates, task-050 smoke, study and judging."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runner.outcome_v2 import verify_frozen_inputs

ROOT = Path(__file__).resolve().parent.parent
MAIN_CONFIG = "configs/phase1.corrected.main24-v1.yaml"
SMOKE_CONFIG = "configs/phase1.corrected.smoke050-v1.yaml"
PILOT_DB = ROOT / "data/phase1/corrected/pilot-v14/runs.sqlite"
PILOT_REJUDGMENT = ROOT / "data/phase1/corrected/pilot-v14/rejudgments/outcome-main24-v3"
RUN_DIR = ROOT / "data/phase1/corrected/main24-v1"
LOG_PATH = RUN_DIR / "supervisor.log"
STATUS_PATH = RUN_DIR / "status.json"
LOCK_PATH = RUN_DIR / "supervisor.lock"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Supervisor:
    def __init__(self) -> None:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, Any] = {
            "experiment": "phase1_corrected_main24_v1",
            "state": "running",
            "stage": "starting",
            "started_at": _now(),
            "last_updated": _now(),
            "log": str(LOG_PATH.relative_to(ROOT)),
        }
        if LOCK_PATH.exists():
            try:
                pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                raise RuntimeError("A main24 supervisor already appears to be running")
            except (ValueError, OSError):
                LOCK_PATH.unlink(missing_ok=True)
        LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        self._status()

    def _status(self) -> None:
        self.status["last_updated"] = _now()
        _write_json(STATUS_PATH, self.status)

    def stage(self, name: str, *args: str) -> None:
        self.status["stage"] = name
        self.status["stage_started_at"] = _now()
        self._status()
        command = [sys.executable, "-m", "runner.cli", *args]
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_now()}] START {name}\n")
            log.write(f"command: {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            last_checkpoint = time.monotonic()
            while process.poll() is None:
                if (
                    name in {"task_050_nine_cell_smoke", "main_216_agent_matrix"}
                    and time.monotonic() - last_checkpoint >= 60
                ):
                    database = (
                        ROOT / "data/phase1/corrected/smoke050-v1/runs.sqlite"
                        if name == "task_050_nine_cell_smoke"
                        else ROOT / "data/phase1/corrected/main24-v1/runs.sqlite"
                    )
                    experiment_id = (
                        "phase1_corrected_smoke050_v1"
                        if name == "task_050_nine_cell_smoke"
                        else "phase1_corrected_main24_v1"
                    )
                    progress = self._progress(database, experiment_id)
                    self.status["cell_progress"] = progress
                    self._status()
                    log.write(f"[{_now()}] CHECKPOINT {json.dumps(progress)}\n")
                    log.flush()
                    last_checkpoint = time.monotonic()
                time.sleep(5)
            return_code = process.wait()
            log.write(f"[{_now()}] END {name} exit={return_code}\n")
        self.status.setdefault("stages", {})[name] = {
            "exit_code": return_code,
            "finished_at": _now(),
        }
        self._status()
        if return_code != 0:
            raise RuntimeError(f"Stage {name} failed; see the local supervisor log")

    @staticmethod
    def _progress(database: Path, experiment_id: str) -> dict[str, int]:
        if not database.is_file():
            return {"cells": 0, "completed": 0, "gradable": 0, "infrastructure_error": 0}
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = con.execute(
                """SELECT count(*),
                          sum(CASE WHEN status='completed' THEN 1 ELSE 0 END),
                          sum(CASE WHEN status='infrastructure_error' THEN 1 ELSE 0 END)
                   FROM run WHERE experiment_id=?""",
                (experiment_id,),
            ).fetchone()
            gradable = con.execute(
                """SELECT count(*) FROM run r JOIN grade g ON g.run_id=r.run_id
                   WHERE r.experiment_id=? AND r.status='completed'
                     AND g.test_name='task_grader' AND g.status='completed'
                     AND g.score IS NOT NULL""",
                (experiment_id,),
            ).fetchone()[0]
        finally:
            con.close()
        return {
            "cells": int(row[0] or 0),
            "completed": int(row[1] or 0),
            "gradable": int(gradable or 0),
            "infrastructure_error": int(row[2] or 0),
        }

    def _verify_pilot_baselines(self) -> None:
        record = verify_frozen_inputs(ROOT, PILOT_DB)
        self.status["preserved_pilot_baselines"] = record
        self._status()

    def _check_cells(self, database: Path, experiment_id: str, expected: int) -> None:
        if not database.is_file():
            raise RuntimeError(f"Expected run database is missing for {experiment_id}")
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                """SELECT r.status,r.total_tokens,r.model_metadata_json,r.trace_path,
                          g.status AS grade_status,
                          g.score,g.details
                   FROM run r LEFT JOIN grade g ON g.run_id=r.run_id
                     AND g.test_name='task_grader' AND g.kind='task_grader'
                   WHERE r.experiment_id=?""",
                (experiment_id,),
            ).fetchall()
        finally:
            con.close()
        if len(rows) != expected:
            raise RuntimeError(f"{experiment_id} has {len(rows)} cells; expected {expected}")
        incomplete = [
            row
            for row in rows
            if row["status"] != "completed"
            or row["grade_status"] != "completed"
            or row["score"] is None
        ]
        if incomplete:
            raise RuntimeError(f"{experiment_id} has {len(incomplete)} missing or ungradable cells")
        if experiment_id.endswith("smoke050_v1"):
            if any(row["total_tokens"] is None for row in rows):
                raise RuntimeError("Task-050 smoke has missing model usage")
            for row in rows:
                metadata = json.loads(row["model_metadata_json"] or "{}")
                if (
                    metadata.get("provider") != "university_gpu"
                    or metadata.get("model") != "phase1-university-model"
                ):
                    raise RuntimeError(
                        "A smoke cell did not route through the university model proxy"
                    )
                trace_path = Path(row["trace_path"] or "")
                trace = [
                    json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
                if not any(event.get("event_type") == "proxy_response" for event in trace):
                    raise RuntimeError("A smoke cell has no successful proxy response event")

    def _check_outcomes(self, database: Path, expected: int) -> None:
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            completed = con.execute(
                "SELECT count(*) FROM judgment "
                "WHERE status='completed' AND outcome_score IS NOT NULL"
            ).fetchone()[0]
            errors = con.execute(
                "SELECT count(*) FROM judgment WHERE status NOT IN ('completed')"
            ).fetchone()[0]
        finally:
            con.close()
        if completed != expected or errors:
            raise RuntimeError(
                f"Hybrid outcome coverage is {completed}/{expected}; unresolved statuses={errors}"
            )

    def _check_process_judgments(self, database: Path, experiment_id: str, expected: int) -> None:
        con = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            counts = con.execute(
                """SELECT pg.status,count(*) FROM run r
                   LEFT JOIN process_grade pg ON pg.run_id=r.run_id
                   WHERE r.experiment_id=? GROUP BY pg.status""",
                (experiment_id,),
            ).fetchall()
        finally:
            con.close()
        total = sum(int(row[1]) for row in counts)
        if total != expected:
            raise RuntimeError(f"Process/security judging represented {total}/{expected} cells")
        self.status.setdefault("judge_warnings", {})[experiment_id] = {
            str(status or "missing"): int(count)
            for status, count in counts
            if status != "completed"
        }
        self._status()

    def run(self) -> None:
        self._verify_pilot_baselines()
        self.stage("full_24_task_preflight", "preflight-study", "--config", MAIN_CONFIG)
        self._verify_pilot_baselines()
        self.stage("llm_judge_calibration", "calibrate-main24-outcomes", "--config", MAIN_CONFIG)
        self._verify_pilot_baselines()
        self.stage(
            "archived_44_run_pilot_backtest",
            "judge-main24-outcomes",
            "--config",
            MAIN_CONFIG,
            "--database",
            "data/phase1/corrected/pilot-v14/runs.sqlite",
            "--output",
            "data/phase1/corrected/pilot-v14/rejudgments/outcome-main24-v3/judgments.sqlite",
            "--expected-cells",
            "44",
            "--experiment-id",
            "phase1_corrected_pilot_v14",
        )
        self._verify_pilot_baselines()

        self.stage(
            "task_050_nine_cell_smoke",
            "run",
            "--config",
            SMOKE_CONFIG,
            "--max-cells",
            "9",
        )
        smoke_db = ROOT / "data/phase1/corrected/smoke050-v1/runs.sqlite"
        smoke_id = "phase1_corrected_smoke050_v1"
        self._check_cells(smoke_db, smoke_id, 9)
        self.stage(
            "task_050_llm_outcome_judging",
            "judge-main24-outcomes",
            "--config",
            MAIN_CONFIG,
            "--database",
            "data/phase1/corrected/smoke050-v1/runs.sqlite",
            "--output",
            "data/phase1/corrected/smoke050-v1/outcomes-v3.sqlite",
            "--expected-cells",
            "9",
            "--experiment-id",
            smoke_id,
        )
        self._check_outcomes(ROOT / "data/phase1/corrected/smoke050-v1/outcomes-v3.sqlite", 9)
        self.stage(
            "task_050_process_security_judging",
            "judge",
            "--config",
            SMOKE_CONFIG,
            "--outcome-database",
            "data/phase1/corrected/smoke050-v1/outcomes-v3.sqlite",
        )
        self._check_process_judgments(smoke_db, smoke_id, 9)

        self.stage("main_216_agent_matrix", "run", "--config", MAIN_CONFIG)
        main_db = ROOT / "data/phase1/corrected/main24-v1/runs.sqlite"
        main_id = "phase1_corrected_main24_v1"
        self._check_cells(main_db, main_id, 216)
        self.stage(
            "main_216_llm_outcome_judging",
            "judge-main24-outcomes",
            "--config",
            MAIN_CONFIG,
        )
        self._check_outcomes(ROOT / "data/phase1/corrected/main24-v1/outcomes-v3.sqlite", 216)
        self.stage(
            "main_216_process_security_judging",
            "judge",
            "--config",
            MAIN_CONFIG,
            "--outcome-database",
            "data/phase1/corrected/main24-v1/outcomes-v3.sqlite",
        )
        self._check_process_judgments(main_db, main_id, 216)
        self._verify_pilot_baselines()
        self.status.update(
            {
                "state": "report_ready",
                "stage": "complete",
                "finished_at": _now(),
                "report_command": (
                    ".venv/Scripts/python -m analysis.main24_report "
                    "--config configs/phase1.corrected.main24-v1.yaml"
                ),
            }
        )
        self._status()
        (RUN_DIR / "REPORT_READY").write_text(_now(), encoding="utf-8")


def main() -> int:
    supervisor = Supervisor()
    try:
        supervisor.run()
        return 0
    except Exception as exc:
        supervisor.status.update(
            {
                "state": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": _now(),
            }
        )
        supervisor._status()
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"[{_now()}] SUPERVISOR FAILED: {type(exc).__name__}: {exc}\n")
        return 2
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
