"""Finalize a fully recorded calibration store after an analysis-only failure."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from runner.outcome_calibration import build_calibration_cases
from runner.outcome_judge import load_rubric, sha256_bytes, sha256_text, utc_now
from runner.outcome_v2 import (
    MAX_CALL_ATTEMPTS,
    OUTCOME_VERSION,
    ROUND_ID,
    OutcomeV2Error,
    _assert_transport_smoke,
    _check_llm_controls,
    _check_oracle_controls,
    _code_manifest,
    _scan_for_secrets,
    _tree_digest,
    verify_frozen_inputs,
)


def finalize_calibration_store(
    *,
    root: Path,
    source_database: Path,
    rubric_path: Path,
    task_root: Path,
    store_path: Path,
    output_path: Path,
    public_model_alias: str,
    model_identity_sha256: str,
    secrets: tuple[str, ...],
    seed: int = 1701,
) -> dict[str, Any]:
    """Build calibration.json from all 29 already-scored, hash-verified controls."""
    baselines = verify_frozen_inputs(root, source_database)
    connection = sqlite3.connect(store_path)
    connection.row_factory = sqlite3.Row
    try:
        manifest_row = connection.execute(
            "SELECT value FROM meta WHERE key='manifest'"
        ).fetchone()
        if manifest_row is None:
            raise OutcomeV2Error("Calibration store lacks its frozen manifest")
        manifest = json.loads(manifest_row[0])
        if (
            manifest.get("outcome_version") != OUTCOME_VERSION
            or manifest.get("round_id") != ROUND_ID
            or manifest.get("source_database_sha256") != baselines["source_database_sha256"]
            or manifest.get("original_pdf_sha256") != baselines["original_pdf_sha256"]
            or manifest.get("rubric_sha256") != sha256_bytes(rubric_path.read_bytes())
            or manifest.get("model_identity_sha256") != model_identity_sha256
            or manifest.get("public_model_alias") != public_model_alias
            or manifest.get("code_sha256") != _code_manifest(root)
        ):
            raise OutcomeV2Error("Calibration store provenance does not match current v7 inputs")

        cases = build_calibration_cases(task_root)
        if len(cases) != 29:
            raise OutcomeV2Error("Expected all 29 frozen calibration cases")
        rows = {
            row["case_id"]: row
            for row in connection.execute("SELECT * FROM calibration_control")
        }
        if set(rows) != {case["case_id"] for case in cases}:
            raise OutcomeV2Error("Calibration store does not contain the exact 29-case set")
        oracle_scores: dict[str, dict[str, Any]] = {}
        scored: dict[str, dict[str, Any]] = {}
        controls_digest_input = []
        for case in cases:
            case_id = case["case_id"]
            row = rows[case_id]
            stored_packet_json = row["packet_json"]
            stored_packet = json.loads(stored_packet_json)
            if sha256_text(stored_packet_json) != row["packet_sha256"]:
                raise OutcomeV2Error(f"Stored calibration packet hash is invalid: {case_id}")

            def stable_packet(value: Any, key: str | None = None) -> Any:
                if isinstance(value, dict):
                    return {name: stable_packet(item, name) for name, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [stable_packet(item, key) for item in value]
                if key == "stdout" and isinstance(value, str):
                    return re.sub(r" in \d+(?:\.\d+)?s", " in <elapsed>s", value)
                return value

            packet_matches = stable_packet(stored_packet) == stable_packet(case["packet"])
            files_hash = _tree_digest(case["files"])
            if (
                row["status"] != "completed"
                or row["score"] is None
                or not packet_matches
                or row["files_sha256"] != files_hash
            ):
                raise OutcomeV2Error(f"Calibration evidence is incomplete or changed: {case_id}")
            oracle_scores[case_id] = {
                "score": float(row["oracle_score"]),
                "details": json.loads(row["oracle_details_json"]),
            }
            ratings = json.loads(row["ratings_json"])
            scored[case_id] = {
                "task_id": case["task_id"],
                "kind": case["kind"],
                "score": float(row["score"]),
                "ratings": ratings,
                "oracle_score": float(row["oracle_score"]),
            }
            controls_digest_input.append(
                {
                    "case_id": case_id,
                    "packet_sha256": row["packet_sha256"],
                    "files_sha256": files_hash,
                }
            )
            attempts = connection.execute(
                "SELECT * FROM call_attempt WHERE subject_id=? AND round_id=? ORDER BY attempt_no",
                (case_id, ROUND_ID),
            ).fetchall()
            if not attempts or len(attempts) > MAX_CALL_ATTEMPTS:
                raise OutcomeV2Error(f"Missing or excessive attempt history for {case_id}")
            if attempts[-1]["status"] not in {"completed", "deterministic"}:
                raise OutcomeV2Error(f"Calibration case has no successful final attempt: {case_id}")
            if any(
                attempt["status"] not in {"completed", "deterministic", "error"}
                for attempt in attempts
            ):
                raise OutcomeV2Error(f"Invalid attempt status for {case_id}")

        # The control checker used underscore lookup keys for equivalent-language
        # controls, while the persistent IDs correctly use hyphens. Add aliases
        # in memory; stored IDs and scored evidence remain unchanged.
        check_input = dict(scored)
        for task_id in ("019-incident-runbook-synthesis", "025-meeting-action-tracker"):
            for language in ("hindi", "hinglish"):
                source_id = f"{task_id}:equivalent-{language}"
                check_input[f"{task_id}:equivalent_{language}"] = scored[source_id]
        checks = _check_llm_controls(check_input)
        oracle_checks = _check_oracle_controls(oracle_scores, cases)
        smoke_id = "transport-smoke:001-file:good"
        smoke_runtime = SimpleNamespace(
            model_config={"model": public_model_alias}, secrets=secrets
        )
        smoke = _assert_transport_smoke(
            connection, smoke_runtime, smoke_id, store_path=store_path
        )
        control_hash = sha256_text(
            json.dumps(controls_digest_input, ensure_ascii=False, sort_keys=True)
        )
        status = (
            "passed"
            if all(item["passed"] for item in checks + oracle_checks)
            else "failed_substantive_controls"
        )
        report = {
            "schema_version": 2,
            "calibration_version": "outcome-v7-controls-29",
            "status": status,
            "outcome_version": OUTCOME_VERSION,
            "round_id": ROUND_ID,
            "rubric_version": load_rubric(rubric_path)["version"],
            "rubric_sha256": manifest["rubric_sha256"],
            "model_identity_sha256": manifest["model_identity_sha256"],
            "model_public_label": manifest["model_public_label"],
            "public_model_alias": manifest["public_model_alias"],
            "source_database_sha256": manifest["source_database_sha256"],
            "original_pdf_sha256": manifest["original_pdf_sha256"],
            "source_task_hashes": manifest["source_task_hashes"],
            "code_sha256": manifest["code_sha256"],
            "evidence_schema_version": manifest["evidence_schema_version"],
            "temperature": manifest["temperature"],
            "top_p": manifest["top_p"],
            "max_tokens": manifest["max_tokens"],
            "max_calls_per_subject": manifest["max_calls_per_subject"],
            "control_count": len(cases),
            "controls_sha256": control_hash,
            "transport_smoke": smoke,
            "structural_canary_case_ids": [
                "001-file:good", "016-code-repair-pytest:good",
                "019-incident-runbook-synthesis:good",
                "025-meeting-action-tracker:prompt_injection",
                "050-multitable-join-analysis:good",
            ],
            "deterministic_oracle_checks": oracle_checks,
            "deterministic_oracle_scores": {
                case_id: value["score"] for case_id, value in oracle_scores.items()
            },
            "checks": checks,
            "results": scored,
            "created_at": utc_now(),
            "finalized_from_append_only_store": True,
        }
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("controls_sha256") != control_hash:
                raise OutcomeV2Error("Existing calibration output differs from stored controls")
            return existing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _scan_for_secrets([store_path, output_path], secrets)
        verify_frozen_inputs(root, source_database)
        return report
    finally:
        connection.close()
