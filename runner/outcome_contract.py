"""Frozen v5 evidence boundaries and deterministic eligibility rules.

Reference answers describe correctness; only submitted files can establish delivery.
The model rates content, while these rules prevent an absent artifact from inheriting
credit from the reference section of the packet.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

ARTIFACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "001-file": {
        "correct_line_count": ("out/linecount.txt",),
        "input_preserved": ("in/input.txt",),
    },
    "016-code-repair-pytest": {
        "working_repair": ("in/app/config_manager.py", "independent_pytest"),
        "constraints_and_test_integrity": ("in/app/config_manager.py", "in/app/test_config.py"),
        "progress_note": ("out/progress.md",),
    },
    "019-incident-runbook-synthesis": {
        "incident_report": ("out/incident_report.json",),
        "rollback_plan": ("out/rollback_plan.md",),
        "evidence_matrix": ("out/evidence_matrix.csv",),
        "status_update": ("out/status_update.md",),
    },
    "025-meeting-action-tracker": {
        "action_table": ("out/action_items.csv",),
        "owner_followups": ("out/owner_followups.md",),
        "merge_rationale": ("out/merge_rationale.md",),
    },
    "050-multitable-join-analysis": {
        "customer_metrics": ("out/customer_metrics.csv",),
        "region_summary": ("out/region_summary.json",),
        "reconciliation_audit": ("out/reconciliation_audit.json",),
        "reconciliation_notes": ("out/reconciliation_notes.md",),
        "input_preserved": ("in/",),
    },
}

REFERENCE_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "001-file": {
        "correct_line_count": ("expected_line_count",),
        "input_preserved": ("source_sha256",),
    },
    "016-code-repair-pytest": {
        "working_repair": ("validation_constraints", "expected_subtask_topics"),
        "constraints_and_test_integrity": ("validation_constraints", "required_files"),
        "progress_note": ("expected_subtask_topics",),
    },
    "019-incident-runbook-synthesis": {
        "incident_report": ("expected", "timeline_min_items", "required_actions_keywords"),
        "rollback_plan": ("required_plan_phrases",),
        "evidence_matrix": ("evidence_required_sources", "evidence_min_items"),
        "status_update": ("required_status_phrases",),
    },
    "025-meeting-action-tracker": {
        "action_table": ("expected_actions", "forbidden_task_contains"),
        "owner_followups": ("owners", "expected_actions"),
        "merge_rationale": ("rationale_terms",),
    },
    "050-multitable-join-analysis": {
        "customer_metrics": ("header", "rows"),
        "region_summary": ("region_summary_expected",),
        "reconciliation_audit": ("audit_expected",),
        "reconciliation_notes": ("required_notes_terms",),
        "input_preserved": ("fixture_hashes",),
    },
}


def reference_answer(task_id: str, source: Path) -> dict[str, Any]:
    ground_truth = source / "ground_truth.json"
    if ground_truth.is_file():
        return {
            "kind": "upstream_reference_facts_and_constraints",
            "source_sha256": sha256(ground_truth.read_bytes()).hexdigest(),
            "facts": json.loads(ground_truth.read_text(encoding="utf-8")),
        }
    if task_id == "001-file":
        fixture = source / "fixtures/in/input.txt"
        raw = fixture.read_bytes()
        return {
            "kind": "computed_from_pinned_input",
            "source_sha256": sha256(raw).hexdigest(),
            "expected_line_count": len(raw.splitlines()),
        }
    raise ValueError(f"No frozen reference answer for {task_id}")


def criterion_contract(
    task_id: str, source: Path, files: dict[str, bytes], validation: Any
) -> dict[str, dict[str, Any]]:
    if task_id not in ARTIFACTS:
        raise ValueError(f"Unmapped outcome task: {task_id}")
    fixture_root = source / "fixtures"
    result: dict[str, dict[str, Any]] = {}
    for criterion, required in ARTIFACTS[task_id].items():
        submitted: list[dict[str, Any]] = []
        for path in required:
            if path == "independent_pytest":
                submitted.append(
                    {"path": path, "present": validation is not None, "result": validation}
                )
            elif path == "in/":
                input_paths = sorted(p for p in files if p.startswith("in/"))
                submitted.extend(
                    {"path": p, "present": bool(files[p]), "sha256": sha256(files[p]).hexdigest()}
                    for p in input_paths
                )
            else:
                content = files.get(path)
                submitted.append(
                    {
                        "path": path,
                        "present": bool(content),
                        "sha256": sha256(content).hexdigest() if content is not None else None,
                    }
                )
        forced: int | None = None
        ceiling = 4
        if any(not entry["present"] for entry in submitted) or not submitted:
            forced = 0
        if criterion == "input_preserved":
            original = {
                p.relative_to(fixture_root).as_posix(): p.read_bytes()
                for p in fixture_root.rglob("*")
                if p.is_file() and p.relative_to(fixture_root).parts[0] == "in"
            }
            forced = 4 if all(files.get(p) == raw for p, raw in original.items()) else 0
        if task_id == "016-code-repair-pytest":
            original_code = (fixture_root / "in/app/config_manager.py").read_bytes()
            original_test = (fixture_root / "in/app/test_config.py").read_bytes()
            if criterion == "working_repair":
                if files.get("in/app/config_manager.py") == original_code:
                    forced = 0
                elif (
                    validation is None
                    or validation.get("timed_out")
                    or validation.get("returncode") != 0
                ):
                    ceiling = 2
            elif (
                criterion == "constraints_and_test_integrity"
                and files.get("in/app/test_config.py") != original_test
            ):
                forced = 0
        result[criterion] = {
            "submitted_evidence": submitted,
            "reference_fields": REFERENCE_FIELDS[task_id][criterion],
            "forced_level": forced,
            "max_level": ceiling,
        }
    return result


def apply_contract(
    ratings: dict[str, dict[str, Any]], packet: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    contract = packet["criterion_contract"]
    result: dict[str, dict[str, Any]] = {}
    if set(ratings) != {key for key, rule in contract.items() if rule["forced_level"] is None}:
        raise ValueError("Judge response criteria differ from criteria requiring model assessment")
    for criterion, rule in contract.items():
        item = ratings.get(criterion)
        forced = rule["forced_level"]
        if forced is None:
            assert item is not None
            model_level: int | None = item["level"]
            level = min(model_level, rule["max_level"])
            rationale = item["rationale"]
            rationale_status = item["rationale_status"]
        else:
            model_level = None
            level = forced
            rationale = "Assigned by the frozen artifact contract; no model rating requested."
            rationale_status = "deterministic"
        result[criterion] = {
            "level": level,
            "model_level": model_level,
            "rationale": rationale,
            "rationale_status": rationale_status,
            "evidence": rule["submitted_evidence"],
            "reference_fields": rule.get("reference_fields", ()),
        }
    return result


def model_response_from_ratings(ratings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct only model-supplied ratings for independent audit validation."""
    return {
        "ratings": {
            criterion: {
                "level": item["model_level"],
                "rationale": (
                    "" if item.get("rationale_status") == "missing" else item["rationale"]
                ),
            }
            for criterion, item in ratings.items()
            if item.get("model_level") is not None
        }
    }
