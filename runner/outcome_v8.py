"""Calibrate and rejudge the immutable v14 pilot with outcome-v8."""

from __future__ import annotations

import csv
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from runner.outcome_calibration import build_calibration_cases
from runner.outcome_judge import load_rubric, load_source_cells, run_code_test_evidence, utc_now
from runner.outcome_v2 import (
    KNOWN_MISSING,
    OutcomeJudgeRuntime,
    _selection_sources,
    verify_frozen_inputs,
)
from runner.outcome_v8_judge import (
    V8JudgeError,
    V8SubstantiveError,
    judge_subject,
    open_store,
    register_subject,
    sha,
)
from runner.outcome_v8_packet import build_comparison_packet, packet_from_archive

TASK_ROOT_RELATIVE = Path("data/phase1/tasks-v13")
SEED = 1701


def _manifest(
    root: Path,
    source_database: Path,
    rubric_path: Path,
    runtime: OutcomeJudgeRuntime,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    frozen = verify_frozen_inputs(root, source_database)
    relevant = (
        "runner/outcome_v8.py",
        "runner/outcome_v8_judge.py",
        "runner/outcome_v8_packet.py",
        "runner/outcome_calibration.py",
        "runner/outcome_judge.py",
        "runner/outcome_v2.py",
        "runner/proxy.py",
        "runner/redaction.py",
        "runner/sandbox.py",
    )
    return {
        "version": "outcome-v8",
        "source": frozen,
        "rubric_sha256": sha(rubric_path.read_bytes()),
        "model_identity_sha256": runtime.model_identity_sha256,
        "model_alias": runtime.model_config["model"],
        "prompt_temperature": 0,
        "prompt_top_p": 1,
        "max_tokens": 8192,
        "seed": SEED,
        "source_hashes": source_hashes,
        "code_sha256": {name: sha((root / name).read_bytes()) for name in relevant},
    }


def _copy_good(task_id: str, cases: list[dict[str, Any]]) -> dict[str, bytes]:
    return dict(next(case["files"] for case in cases if case["case_id"] == f"{task_id}:good"))


def _csv_mutate(raw: bytes, mutate: Any) -> bytes:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows = list(reader)
    headers = list(reader.fieldnames or [])
    rows = mutate(rows)
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def calibration_cases(task_root: Path, rubric: dict[str, Any], image: str) -> list[dict[str, Any]]:
    original = build_calibration_cases(task_root)
    result = []
    for case in original:
        validation = case["packet"].get("independent_validation")
        packet = build_comparison_packet(
            case["task_id"],
            task_root / case["task_id"] / "source",
            case["files"],
            rubric,
            validation,
        )
        result.append(
            {
                "id": case["case_id"],
                "task_id": case["task_id"],
                "kind": case["kind"],
                "packet": packet,
            }
        )
    for task_id in rubric["tasks"]:
        good = _copy_good(task_id, original)
        near = dict(good)
        alternative = dict(good)
        if task_id == "001-file":
            near["out/linecount.txt"] = b"999\n"
            alternative["out/linecount.txt"] = b"4"
        elif task_id == "016-code-repair-pytest":
            near["in/app/config_manager.py"] = (
                task_root / task_id / "source/fixtures/in/app/config_manager.py"
            ).read_bytes()
            near["out/progress.md"] = b"A careful repair was completed and verified.\n"
            alternative["in/app/config_manager.py"] = (
                b"def deep_update(base_dict, update_dict):\n"
                b"    for key, value in update_dict.items():\n"
                b"        if isinstance(value, dict) and isinstance(base_dict.get(key), dict):\n"
                b"            base_dict[key] = deep_update(base_dict[key], value)\n"
                b"        else:\n"
                b"            base_dict[key] = value\n"
                b"    return base_dict\n"
            )
        elif task_id == "019-incident-runbook-synthesis":
            report = json.loads(near["out/incident_report.json"])
            report["root_cause_service"] = "payment-api"
            near["out/incident_report.json"] = json.dumps(report, indent=2).encode()
            alternative["out/status_update.md"] = (
                b"APAC checkout, payment and inventory are affected. "
                b"SEV2 investigation remains active. No production change has been made. "
                b"Obtain approval before a scoped rollback, verify "
                b"checkout and reserveStock, and update stakeholders in 15 minutes.\n"
            )
        elif task_id == "025-meeting-action-tracker":
            near["out/action_items.csv"] = _csv_mutate(
                near["out/action_items.csv"],
                lambda rows: [
                    {**row, "owner": "Wrong Owner", "status": "done"}
                    if row.get("action_id") == "AT-101"
                    else row
                    for row in rows
                ],
            )
            alternative["out/owner_followups.md"] = (
                b"Rita: pricing FAQ due 2026-06-07. Owen: sandbox access checklist due 2026-06-13. "
                b"Sam: compliance and security work due 2026-06-12 and 2026-06-13. "
                b"Mina: customer quotes due 2026-06-11. Alex: AT-106 waits on its dependency.\n"
            )
        elif task_id == "050-multitable-join-analysis":

            def damage(rows: list[dict[str, str]]) -> list[dict[str, str]]:
                return [
                    {**row, "net_revenue_usd": "9999.00"}
                    if row.get("canonical_customer_id") == "C003"
                    else row
                    for row in rows
                    if row.get("canonical_customer_id") != "C001"
                ]

            near["out/customer_metrics.csv"] = _csv_mutate(near["out/customer_metrics.csv"], damage)
            alternative["out/reconciliation_notes.md"] = (
                b"Use captured payments for revenue. Ignore duplicate captures under the "
                b"documented tuple rule; identify orphan captures and refunds on cancelled "
                b"orders. Exclude chargebacks CB7003/CB7004 from deductions. "
                b"Merge aliases C006, C007 and C010 into their canonical customers.\n"
            )
        for kind, files in (("near_miss", near), ("valid_alternative", alternative)):
            validation = (
                run_code_test_evidence(files, image)
                if task_id == "016-code-repair-pytest"
                else None
            )
            packet = build_comparison_packet(
                task_id, task_root / task_id / "source", files, rubric, validation
            )
            result.append(
                {"id": f"{task_id}:{kind}", "task_id": task_id, "kind": kind, "packet": packet}
            )
    if len(result) != 39 or len({case["id"] for case in result}) != 39:
        raise V8JudgeError("V8 calibration must contain exactly 39 distinct controls")
    return result


def _control_checks(
    cases: list[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    checks = []

    def add(name: str, passed: bool, observed: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed})

    for task_id in sorted({case["task_id"] for case in cases}):
        scores = {
            case["kind"]: results[case["id"]]["score"]
            for case in cases
            if case["task_id"] == task_id
        }
        if any(value is None for value in scores.values()):
            add(f"{task_id}:all controls rated", False, scores)
            continue
        add(f"{task_id}:good >= 0.90", scores["good"] >= 0.9, scores)
        add(f"{task_id}:missing below good", scores["missing"] < scores["good"], scores)
        add(f"{task_id}:wrong below good", scores["incorrect"] < scores["good"], scores)
        add(f"{task_id}:near miss below good", scores["near_miss"] < scores["good"], scores)
        add(
            f"{task_id}:alternative within 0.10",
            abs(scores["valid_alternative"] - scores["good"]) <= 0.1,
            scores,
        )
        add(
            f"{task_id}:injection change <= 0.10",
            abs(scores["prompt_injection"] - scores["good"]) <= 0.1,
            scores,
        )
        for language in ("hindi", "hinglish"):
            key = f"equivalent_{language}"
            if key in scores:
                add(
                    f"{task_id}:{language} equivalent <= 0.10",
                    abs(scores[key] - scores["good"]) <= 0.1,
                    scores,
                )
    # The failed v7 exact and missing controls are decisive in v8.
    one = results.get("001-file:incorrect", {}).get("ratings", {})
    add(
        "001 wrong line count receives zero",
        one.get("correct_line_count.value", {}).get("level") == 0,
        one.get("correct_line_count.value"),
    )
    for task_id in sorted({case["task_id"] for case in cases}):
        packet = next(case["packet"] for case in cases if case["id"] == f"{task_id}:missing")
        ratings = results[f"{task_id}:missing"].get("ratings", {})
        absent = {
            item["id"]
            for item in packet["submitted_answer"]["observations"]
            if item["observed"] is None
        }
        add(
            f"{task_id}:absent deliverables zero",
            all(ratings.get(key, {}).get("level") == 0 for key in absent),
            {key: ratings.get(key, {}).get("level") for key in absent},
        )
    return checks


def _make_pack(
    root: Path,
    source_database: Path,
    task_root: Path,
    experiment_id: str,
    rubric: dict[str, Any],
    image: str,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    verify_frozen_inputs(root, source_database)
    cells = load_source_cells(source_database, experiment_id)
    if len(cells) != 45 or sum(cell.status == "completed" for cell in cells) != 44:
        raise V8JudgeError("Frozen v14 pilot must contain 44 completed and one missing cell")
    missing = [cell for cell in cells if (cell.task_id, cell.language, cell.agent) == KNOWN_MISSING]
    if len(missing) != 1 or missing[0].status != "infrastructure_error":
        raise V8JudgeError("Known v14 missing cell differs from the frozen record")
    packets = {}
    for cell in cells:
        if cell.status == "completed":
            packet, archive_hash, workspace_hash = packet_from_archive(
                cell, task_root, rubric, image
            )
            packets[cell.run_id] = {
                "packet": packet,
                "archive_sha256": archive_hash,
                "workspace_sha256": workspace_hash,
            }
    verify_frozen_inputs(root, source_database)
    return cells, packets


def calibrate(
    root: Path,
    source_database: Path,
    config: dict[str, Any],
    rubric_path: Path,
    output: Path,
    runtime: OutcomeJudgeRuntime,
) -> dict[str, Any]:
    task_root = root / TASK_ROOT_RELATIVE
    rubric = load_rubric(rubric_path)
    sources = _selection_sources(root, task_root, config)
    cells, packets = _make_pack(
        root,
        source_database,
        task_root,
        "phase1_corrected_pilot_v14",
        rubric,
        config["sandbox"]["image"],
    )
    cases = calibration_cases(task_root, rubric, config["sandbox"]["image"])
    manifest = _manifest(root, source_database, rubric_path, runtime, sources)
    store_path = output / "calibration.sqlite"
    db = open_store(store_path, manifest)
    try:
        for case in cases:
            register_subject(db, case["id"], "calibration", case["task_id"], case["packet"])
        # Exercise all five evidence types before spending GPU calls on variants.
        canaries = [case for case in cases if case["kind"] == "good"]
        ordered = [*canaries, *[case for case in cases if case not in canaries]]
        results = {}
        for case in ordered:
            try:
                results[case["id"]] = judge_subject(
                    runtime, db, case["id"], case["packet"], verify=True
                )
            except V8SubstantiveError as exc:
                db.execute(
                    "UPDATE subject SET status='needs_review',reason=?,updated_at=? "
                    "WHERE subject_id=?",
                    (str(exc), utc_now(), case["id"]),
                )
                db.commit()
                results[case["id"]] = {
                    "status": "needs_review", "score": None, "ratings": {},
                }
            except V8JudgeError as exc:
                db.execute(
                    "UPDATE subject SET status='judge_error',reason=?,updated_at=? "
                    "WHERE subject_id=?",
                    (str(exc), utc_now(), case["id"]),
                )
                db.commit()
                raise
        checks = _control_checks(cases, results)
        summary = {
            "version": "outcome-v8",
            "manifest": manifest,
            "control_count": len(cases),
            "checks": checks,
            "checks_failed": sum(not item["passed"] for item in checks),
            "status": "passed"
            if all(item["passed"] for item in checks)
            else "failed_substantive_controls",
            "cases": {
                key: {
                    "score": value["score"],
                    "status": value["status"],
                    "ratings": value["ratings"],
                }
                for key, value in results.items()
            },
            "archived_runs_preflighted": len(packets),
            "source_cells": len(cells),
        }
        output.mkdir(parents=True, exist_ok=True)
        target = output / "calibration.json"
        serialized = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if target.exists() and target.read_text(encoding="utf-8") != serialized:
            raise V8JudgeError(
                "Stored v8 calibration report differs; keep the audit record and version changes"
            )
        if not target.exists():
            target.write_text(serialized, encoding="utf-8")
        return summary
    finally:
        db.close()
        verify_frozen_inputs(root, source_database)


def rejudge(
    root: Path,
    source_database: Path,
    config: dict[str, Any],
    rubric_path: Path,
    output: Path,
    runtime: OutcomeJudgeRuntime,
) -> dict[str, Any]:
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    task_root = root / TASK_ROOT_RELATIVE
    rubric = load_rubric(rubric_path)
    sources = _selection_sources(root, task_root, config)
    manifest = _manifest(root, source_database, rubric_path, runtime, sources)
    if calibration.get("manifest") != manifest or calibration.get("control_count") != 39:
        raise V8JudgeError("V8 calibration identity or coverage differs from judging")
    cells, packets = _make_pack(
        root,
        source_database,
        task_root,
        "phase1_corrected_pilot_v14",
        rubric,
        config["sandbox"]["image"],
    )
    db = open_store(output / "judgments.sqlite", manifest)
    try:
        for cell in cells:
            if cell.run_id in packets:
                saved = packets[cell.run_id]
                register_subject(
                    db,
                    cell.run_id,
                    "pilot",
                    cell.task_id,
                    saved["packet"],
                    saved["archive_sha256"],
                    saved["workspace_sha256"],
                )
            else:
                db.execute(
                    "INSERT OR IGNORE INTO subject VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cell.run_id,
                        "pilot",
                        cell.task_id,
                        "missing_agent_artifact",
                        None,
                        None,
                        "missing_agent_artifact",
                        None,
                        None,
                        "No archived agent workspace",
                        utc_now(),
                    ),
                )
        db.commit()
        order = [cell for cell in cells if cell.run_id in packets]
        random.Random(SEED).shuffle(order)
        for cell in order:
            try:
                judge_subject(runtime, db, cell.run_id, packets[cell.run_id]["packet"], verify=True)
            except V8SubstantiveError as exc:
                db.execute(
                    "UPDATE subject SET status='needs_review',reason=?,updated_at=? "
                    "WHERE subject_id=?",
                    (str(exc), utc_now(), cell.run_id),
                )
                db.commit()
            except V8JudgeError as exc:
                db.execute(
                    "UPDATE subject SET status='judge_error',reason=?,updated_at=? "
                    "WHERE subject_id=?",
                    (str(exc), utc_now(), cell.run_id),
                )
                db.commit()
        statuses = Counter(row[0] for row in db.execute("SELECT status FROM subject"))
        return {
            "version": "outcome-v8",
            "statuses": dict(statuses),
            "order_seed": SEED,
            "calibration_status": calibration["status"],
        }
    finally:
        db.close()
        verify_frozen_inputs(root, source_database)
