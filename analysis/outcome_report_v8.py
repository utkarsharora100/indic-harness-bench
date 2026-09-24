"""Independent, read-only tables and provisional PDF for frozen outcome-v8."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.corrected import bootstrap_paired_deltas, paired_outcome_deltas, paired_task_counts
from runner.outcome_judge import load_rubric
from runner.outcome_v2 import _scan_for_secrets, verify_frozen_inputs
from runner.outcome_v8 import _make_pack
from runner.outcome_v8_judge import packet_hash, weighted_score


class V8ReportError(RuntimeError):
    pass


def _read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise V8ReportError(f"Missing outcome store: {path.name}")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise V8ReportError("No rows for report table")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _v7_scores(root: Path) -> dict[str, float | None]:
    path = root / "data/phase1/corrected/pilot-v14/rejudgments/outcome-v7/judgments.sqlite"
    if not path.is_file():
        return {}
    db = _read_only(path)
    try:
        return {
            row["run_id"]: row["score"]
            for row in db.execute("SELECT run_id,score FROM subject_cell")
        }
    finally:
        db.close()


def _write_review_sample(
    output_dir: Path, rows: list[dict[str, Any]], packets: dict[str, dict[str, Any]]
) -> None:
    import random

    eligible = [
        row for row in rows
        if row["status"] in {"completed", "needs_review"}
        and row["workspace_archive"]
    ]
    scored = sorted(
        (
            (abs(float(row["outcome_score"]) - float(row["oracle_score"])), row["run_id"])
            for row in eligible
            if row["outcome_score"] is not None and row["oracle_score"] is not None
        ),
        key=lambda item: item[0],
    )
    disagreement_band = {
        run_id: ("low", "middle", "high")[min(2, index * 3 // max(1, len(scored)))]
        for index, (_delta, run_id) in enumerate(scored)
    }
    # Needs-review cells remain eligible for human assessment, but cannot be
    # assigned an LLM-vs-oracle disagreement band because they have no final score.
    for row in eligible:
        row["disagreement_band"] = disagreement_band.get(row["run_id"], "score_unavailable")
    rng = random.Random(1701)
    candidates = sorted(eligible, key=lambda _row: rng.random())
    tasks = sorted({row["task_id"] for row in candidates})
    languages = ("english", "hindi", "hinglish")
    harnesses = ("react", "nanobot", "openclaw")
    selected: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    harness_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    nodes = 0

    def feasible(start: int) -> bool:
        remaining = candidates[start:]
        if any(language_counts[key] > 4 for key in languages):
            return False
        if any(harness_counts[key] > 4 for key in harnesses):
            return False
        if any(
            language_counts[key] + sum(row["language"] == key for row in remaining) < 4
            for key in languages
        ):
            return False
        if any(
            harness_counts[key] + sum(row["harness"] == key for row in remaining) < 4
            for key in harnesses
        ):
            return False
        if any(
            task_counts[key] + sum(row["task_id"] == key for row in remaining) < 2
            for key in tasks
        ):
            return False
        return all(
            band_counts[band] or any(row["disagreement_band"] == band for row in remaining)
            for band in ("low", "middle", "high")
        )

    def search(start: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > 1_000_000 or len(selected) > 12 or not feasible(start):
            return False
        if len(selected) == 12:
            return (
                all(language_counts[key] == 4 for key in languages)
                and all(harness_counts[key] == 4 for key in harnesses)
                and all(task_counts[key] >= 2 for key in tasks)
                and all(band_counts[key] >= 1 for key in ("low", "middle", "high"))
            )
        if len(candidates) - start < 12 - len(selected):
            return False
        for index in range(start, len(candidates)):
            row = candidates[index]
            if language_counts[row["language"]] >= 4 or harness_counts[row["harness"]] >= 4:
                continue
            selected.append(row)
            language_counts[row["language"]] += 1
            harness_counts[row["harness"]] += 1
            task_counts[row["task_id"]] += 1
            band_counts[row["disagreement_band"]] += 1
            if search(index + 1):
                return True
            selected.pop()
            language_counts[row["language"]] -= 1
            harness_counts[row["harness"]] -= 1
            task_counts[row["task_id"]] -= 1
            band_counts[row["disagreement_band"]] -= 1
        return False

    if not search(0):
        raise V8ReportError("Cannot satisfy deterministic 12-cell human-review quotas")
    packet_rows = []
    key_rows = []
    forms = []
    for index, row in enumerate(selected, 1):
        review_id = f"v8-blind-{index:02d}"
        packet = packets[row["run_id"]]["packet"]
        packet_rows.append(
            {
                "review_id": review_id,
                "canonical_question": packet["canonical_question"],
                "reference_requirements": packet["reference_requirements"],
                "submitted_answer": packet["submitted_answer"],
                "criterion_weights": packet["criterion_weights"],
            }
        )
        key_rows.append({
            "review_id": review_id,
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "language": row["language"],
            "harness": row["harness"],
            "llm_score": row["outcome_score"],
            "oracle_score": row["oracle_score"],
            "disagreement_band": row["disagreement_band"],
        })
        for requirement in packet["reference_requirements"]:
            forms.append(
                {"review_id": review_id, "task_id": row["task_id"],
                 "requirement_id": requirement["id"], "level_0_to_4": "", "notes": ""}
            )
    (output_dir / "human-review-blinded.json").write_text(
        json.dumps(packet_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "human-review-unblinding-key.json").write_text(
        json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for reviewer in (1, 2):
        _write_csv(output_dir / f"human-reviewer-{reviewer}.csv", forms)


def build_v8_tables(
    root: Path,
    source_database: Path,
    judge_database: Path,
    calibration_path: Path,
    rubric_path: Path,
    output_dir: Path,
    image: str,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    """Require complete technical coverage; never invent missing scores."""
    verify_frozen_inputs(root, source_database)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("version") != "outcome-v8" or calibration.get("control_count") != 39:
        raise V8ReportError("V8 calibration is absent or incomplete")
    if calibration.get("status") not in {"passed", "failed_substantive_controls"}:
        raise V8ReportError("V8 calibration has a technical failure")
    rubric = load_rubric(rubric_path)
    cells, packets = _make_pack(
        root,
        source_database,
        root / "data/phase1/tasks-v13",
        "phase1_corrected_pilot_v14",
        rubric,
        image,
    )
    db = _read_only(judge_database)
    try:
        manifest_row = db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if not manifest_row or json.loads(manifest_row[0]) != calibration["manifest"]:
            raise V8ReportError("Calibration and judge identities differ")
        subjects = {row["subject_id"]: dict(row) for row in db.execute("SELECT * FROM subject")}
        if set(subjects) != {cell.run_id for cell in cells}:
            raise V8ReportError("The v8 store must index all 45 source cells")
        statuses = Counter(row["status"] for row in subjects.values())
        if (
            statuses.get("missing_agent_artifact") != 1
            or statuses.get("completed", 0)
            + statuses.get("needs_review", 0)
            + statuses.get("judge_error", 0)
            != 44
        ):
            raise V8ReportError(f"V8 judgment coverage incomplete: {dict(statuses)}")
        calls = [dict(row) for row in db.execute("SELECT * FROM call_attempt")]
        proxy_events = db.execute("SELECT COUNT(*) FROM proxy_event").fetchone()[0]
    finally:
        db.close()
    v7 = _v7_scores(root)
    by_subject_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_subject_calls[call["subject_id"]].append(call)
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for cell in cells:
        subject = subjects[cell.run_id]
        packet = packets.get(cell.run_id, {}).get("packet")
        ratings = json.loads(subject["ratings_json"] or "{}")
        if packet is not None:
            if subject["packet_sha256"] != packet_hash(packet):
                raise V8ReportError("Stored packet identity differs from archived evidence")
            if subject["archive_sha256"] != packets[cell.run_id]["archive_sha256"]:
                raise V8ReportError("Stored archive hash differs")
            if subject["workspace_sha256"] != packets[cell.run_id]["workspace_sha256"]:
                raise V8ReportError("Stored workspace hash differs")
            expected_ids = {item["id"] for item in packet["reference_requirements"]}
            if subject["status"] == "completed" and set(ratings) != expected_ids:
                raise V8ReportError("A finished cell has incomplete criterion ratings")
            if subject["status"] == "needs_review" and ratings and set(ratings) != expected_ids:
                raise V8ReportError("A review-needed cell has partial criterion ratings")
            if subject["status"] == "needs_review" and not ratings and not subject["reason"]:
                raise V8ReportError("A review-needed cell lacks an explicit reason")
            if (
                subject["status"] == "completed"
                and abs(weighted_score(packet, ratings) - subject["score"]) > 1e-6
            ):
                raise V8ReportError("Persisted score differs from independently computed score")
            for requirement in packet["reference_requirements"]:
                item = ratings.get(requirement["id"], {})
                checks.append(
                    {
                        "run_id": cell.run_id,
                        "task_id": cell.task_id,
                        "harness": cell.agent,
                        "language": cell.language,
                        "criterion": requirement["criterion"],
                        "requirement_id": requirement["id"],
                        "kind": requirement["kind"],
                        "expected": json.dumps(requirement["expected"], ensure_ascii=False),
                        "level": item.get("level"),
                        "evidence_id": item.get("evidence_id"),
                        "observation": item.get("observation"),
                    }
                )
        attempts = by_subject_calls.get(cell.run_id, [])
        if subject["status"] in {"completed", "needs_review"} and not attempts:
            raise V8ReportError("Completed cell lacks model-call history")
        input_used = [x["input_tokens"] for x in attempts if x["input_tokens"] is not None]
        output_used = [x["output_tokens"] for x in attempts if x["output_tokens"] is not None]
        rows.append(
            {
                "run_id": cell.run_id,
                "task_id": cell.task_id,
                "harness": cell.agent,
                "agent": cell.agent,
                "language": cell.language,
                "status": subject["status"],
                "grade_status": subject["status"],
                "outcome_score": subject["score"],
                "oracle_score": cell.oracle_score,
                "v7_score": v7.get(cell.run_id),
                "process_score": cell.process.get("process_score"),
                "security_score": cell.process.get("security_score"),
                "judge_calls": len(attempts),
                "judge_error_calls": sum(item["status"] == "error" for item in attempts),
                "judge_input_tokens": sum(input_used) if input_used else None,
                "judge_output_tokens": sum(output_used) if output_used else None,
                "trace": str(cell.trace_path),
                "workspace_archive": str(cell.archive_path) if cell.archive_path else "",
            }
        )
    means = {
        f"{harness}:{language}": mean(
            row["outcome_score"]
            for row in rows
            if row["harness"] == harness
            and row["language"] == language
            and row["outcome_score"] is not None
        )
        for harness in ("react", "nanobot", "openclaw")
        for language in ("english", "hindi", "hinglish")
        if any(
            row["outcome_score"] is not None
            for row in rows
            if row["harness"] == harness and row["language"] == language
        )
    }
    mean_counts = {
        f"{harness}:{language}": len(
            {
                row["task_id"]
                for row in rows
                if row["harness"] == harness
                and row["language"] == language
                and row["outcome_score"] is not None
            }
        )
        for harness in ("react", "nanobot", "openclaw")
        for language in ("english", "hindi", "hinglish")
    }
    paired = paired_outcome_deltas(rows)
    counts = paired_task_counts(rows)
    bootstrap = bootstrap_paired_deltas(rows, repetitions=20_000, seed=1701)
    oracle_diffs = [
        abs(row["outcome_score"] - row["oracle_score"])
        for row in rows
        if row["outcome_score"] is not None and row["oracle_score"] is not None
    ]
    v7_diffs = [
        abs(row["outcome_score"] - row["v7_score"])
        for row in rows
        if row["outcome_score"] is not None and row["v7_score"] is not None
    ]
    result = {
        "version": "outcome-v8",
        "coverage": dict(statuses),
        "calibration_status": calibration["status"],
        "calibration_failed_checks": [x for x in calibration["checks"] if not x["passed"]],
        "calibration_cases": {
            key: {"score": value["score"], "status": value["status"]}
            for key, value in calibration["cases"].items()
        },
        "means": means,
        "mean_task_counts": mean_counts,
        "paired_differences": paired,
        "paired_task_counts": counts,
        "paired_bootstrap": bootstrap,
        "proxy_events": proxy_events,
        "judge_calls": len(calls),
        "judge_error_calls": sum(call["status"] == "error" for call in calls),
        "oracle_comparison": {
            "paired_cells": len(oracle_diffs),
            "mean_absolute_difference": mean(oracle_diffs) if oracle_diffs else None,
        },
        "v7_comparison": {
            "paired_cells": len(v7_diffs),
            "mean_absolute_difference": mean(v7_diffs) if v7_diffs else None,
        },
        "v7_comparison_available": bool(v7),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cells.csv", rows)
    _write_csv(output_dir / "requirements.csv", checks)
    _write_review_sample(output_dir, rows, packets)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _scan_for_secrets(list(output_dir.iterdir()), secrets)
    verify_frozen_inputs(root, source_database)
    return {**result, "rows": rows, "requirements": checks}
