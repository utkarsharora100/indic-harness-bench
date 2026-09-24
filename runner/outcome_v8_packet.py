"""Reference-guided, submission-only comparison packets for outcome-v8.

This module prepares observations; it does not assign grades. The judge sees
the canonical question, frozen reference facts and the agent's final artifacts.
"""

from __future__ import annotations

import csv
import io
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from runner.outcome_contract import ARTIFACTS, reference_answer
from runner.outcome_judge import SourceCell, read_workspace_archive, run_code_test_evidence
from runner.outcome_v8_judge import impossible_credit

SCHEMA_VERSION = 8
MAX_PACKET_BYTES = 400_000


class ComparisonPacketError(ValueError):
    pass


def _decode(content: bytes | None) -> str | None:
    if content is None:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComparisonPacketError("A required submitted artifact is not UTF-8") from exc


def _parsed(files: dict[str, bytes], path: str) -> tuple[Any, str | None]:
    raw = _decode(files.get(path))
    if raw is None:
        return None, "missing artifact"
    try:
        if path.endswith(".json"):
            return json.loads(raw), None
        if path.endswith(".csv"):
            return list(csv.DictReader(io.StringIO(raw))), None
        return raw, None
    except (ValueError, csv.Error) as exc:
        return raw, f"parse error: {type(exc).__name__}: {exc}"


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else None


def _stable_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Retain test evidence while removing nondeterministic pytest elapsed times."""
    stable = dict(validation)
    for field in ("stdout", "stderr"):
        content = stable.get(field)
        if isinstance(content, str):
            stable[field] = re.sub(r"\bin \d+(?:\.\d+)?s\b", "in <elapsed>", content)
    return stable


def build_comparison_packet(
    task_id: str,
    source: Path,
    files: dict[str, bytes],
    rubric: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task_id not in ARTIFACTS or task_id not in rubric["tasks"]:
        raise ComparisonPacketError(f"No frozen v8 comparison contract for {task_id}")
    question = (source / "prompt.txt").read_text(encoding="utf-8")
    gold = reference_answer(task_id, source)
    facts = gold.get("facts", gold)
    criteria = rubric["tasks"][task_id]["criteria"]
    known_paths = sorted(
        {
            path
            for required in ARTIFACTS[task_id].values()
            for path in required
            if path != "independent_pytest"
        }
    )
    artifacts: dict[str, dict[str, Any]] = {}
    path_to_id: dict[str, str] = {}
    for path in known_paths:
        if path == "in/":
            continue
        # Input data tables are never sent to the judge. Code and fixture hashes
        # are observations needed for the code-repair and preservation criteria.
        if path.startswith("in/") and task_id != "016-code-repair-pytest":
            continue
        artifact_id = f"A{len(artifacts) + 1:02d}"
        path_to_id[path] = artifact_id
        parsed, error = _parsed(files, path)
        content = files.get(path)
        artifacts[artifact_id] = {
            "path": path,
            "present": content is not None,
            "sha256": sha256(content).hexdigest() if content is not None else None,
            "content": parsed,
            "parse_error": error,
        }
    if validation is not None:
        validation = _stable_validation(validation)
        path_to_id["independent_pytest"] = "T01"
        artifacts["T01"] = {
            "path": "validation/pytest.txt",
            "present": True,
            "content": validation,
            "parse_error": None,
        }

    checks: list[dict[str, Any]] = []

    def add(
        criterion: str,
        key: str,
        kind: str,
        expected: Any,
        observed: Any,
        path: str,
        *,
        alternatives: list[Any] | None = None,
    ) -> None:
        if criterion not in {item["id"] for item in criteria}:
            raise ComparisonPacketError(f"Unknown rubric criterion: {criterion}")
        artifact_id = path_to_id.get(path)
        if artifact_id is None and path != "fixture_hashes":
            raise ComparisonPacketError(f"Unmapped submitted evidence: {path}")
        checks.append(
            {
                "id": f"{criterion}.{key}",
                "criterion": criterion,
                "kind": kind,
                "expected": expected,
                "observed": observed,
                "submission_evidence_id": artifact_id or "H01",
                "accepted_alternatives": alternatives or [],
            }
        )

    def observed(path: str) -> Any:
        return artifacts[path_to_id[path]]["content"]

    if task_id == "001-file":
        add(
            "correct_line_count",
            "value",
            "exact",
            facts["expected_line_count"],
            observed("out/linecount.txt"),
            "out/linecount.txt",
        )
        original = (source / "fixtures/in/input.txt").read_bytes()
        add(
            "input_preserved",
            "sha256",
            "exact",
            sha256(original).hexdigest(),
            sha256(files["in/input.txt"]).hexdigest() if "in/input.txt" in files else None,
            "fixture_hashes",
        )

    elif task_id == "016-code-repair-pytest":
        code = observed("in/app/config_manager.py")
        test = files.get("in/app/test_config.py")
        add(
            "working_repair",
            "public_tests",
            "exact",
            "pytest passes",
            {
                "returncode": validation.get("returncode") if validation else None,
                "timed_out": validation.get("timed_out") if validation else None,
            },
            "independent_pytest",
        )
        add(
            "working_repair",
            "implementation",
            "semantic",
            facts.get("validation_constraints"),
            code,
            "in/app/config_manager.py",
        )
        original_test = (source / "fixtures/in/app/test_config.py").read_bytes()
        add(
            "constraints_and_test_integrity",
            "test_hash",
            "exact",
            sha256(original_test).hexdigest(),
            sha256(test).hexdigest() if test else None,
            "in/app/test_config.py",
        )
        add(
            "constraints_and_test_integrity",
            "constraints",
            "semantic",
            facts.get("validation_constraints"),
            code,
            "in/app/config_manager.py",
        )
        for index, topic in enumerate(facts.get("expected_subtask_topics", []), 1):
            add(
                "progress_note",
                f"topic_{index:02d}",
                "semantic",
                topic,
                observed("out/progress.md"),
                "out/progress.md",
            )

    elif task_id == "019-incident-runbook-synthesis":
        report = observed("out/incident_report.json")
        for key, expected in sorted(facts["expected"].items()):
            kind = (
                "exact"
                if isinstance(expected, (int, float))
                or key in {"severity", "root_cause_service", "primary_change_id"}
                else "semantic"
            )
            add(
                "incident_report",
                key,
                kind,
                expected,
                _field(report, "blast_radius")
                if key == "blast_radius_keywords"
                else _field(report, key),
                "out/incident_report.json",
            )
        for index, expected in enumerate(facts.get("required_actions_keywords", []), 1):
            add(
                "incident_report",
                f"action_{index:02d}",
                "semantic",
                expected,
                _field(report, "recommended_actions"),
                "out/incident_report.json",
            )
        for index, expected in enumerate(facts.get("required_plan_phrases", []), 1):
            add(
                "rollback_plan",
                f"step_{index:02d}",
                "semantic",
                expected,
                observed("out/rollback_plan.md"),
                "out/rollback_plan.md",
            )
        matrix = observed("out/evidence_matrix.csv")
        for index, expected in enumerate(facts.get("evidence_required_sources", []), 1):
            add(
                "evidence_matrix",
                f"source_{index:02d}",
                "semantic",
                expected,
                matrix,
                "out/evidence_matrix.csv",
            )
        for index, expected in enumerate(facts.get("required_status_phrases", []), 1):
            add(
                "status_update",
                f"point_{index:02d}",
                "semantic",
                expected,
                observed("out/status_update.md"),
                "out/status_update.md",
            )

    elif task_id == "025-meeting-action-tracker":
        submitted_rows = observed("out/action_items.csv")
        by_id = (
            {row.get("action_id"): row for row in submitted_rows if isinstance(row, dict)}
            if isinstance(submitted_rows, list)
            else {}
        )
        for expected in facts["expected_actions"]:
            action_id = expected["action_id"]
            actual = by_id.get(action_id)
            for field in ("owner", "deadline", "status"):
                add(
                    "action_table",
                    f"{action_id}_{field}",
                    "exact",
                    expected[field],
                    _field(actual, field),
                    "out/action_items.csv",
                )
            for field, expected_field in (
                ("task", expected["task_contains"]),
                ("source", expected["source_matches"]),
            ):
                add(
                    "action_table",
                    f"{action_id}_{field}",
                    "semantic",
                    expected_field,
                    _field(actual, field),
                    "out/action_items.csv",
                )
        for index, forbidden in enumerate(facts.get("forbidden_task_contains", []), 1):
            add(
                "action_table",
                f"excluded_{index:02d}",
                "semantic",
                f"Completed or cancelled action absent: {forbidden}",
                submitted_rows,
                "out/action_items.csv",
            )
        for index, owner in enumerate(facts.get("owners", []), 1):
            add(
                "owner_followups",
                f"owner_{index:02d}",
                "semantic",
                owner,
                observed("out/owner_followups.md"),
                "out/owner_followups.md",
            )
        for index, term in enumerate(facts.get("rationale_terms", []), 1):
            add(
                "merge_rationale",
                f"decision_{index:02d}",
                "semantic",
                term,
                observed("out/merge_rationale.md"),
                "out/merge_rationale.md",
            )

    elif task_id == "050-multitable-join-analysis":
        rows = observed("out/customer_metrics.csv")
        by_customer = (
            {row.get("canonical_customer_id"): row for row in rows if isinstance(row, dict)}
            if isinstance(rows, list)
            else {}
        )
        add(
            "customer_metrics",
            "headers",
            "exact",
            facts["header"],
            list(rows[0]) if isinstance(rows, list) and rows else None,
            "out/customer_metrics.csv",
        )
        add(
            "customer_metrics",
            "customer_ids",
            "exact",
            sorted(row["canonical_customer_id"] for row in facts["rows"]),
            sorted(by_customer),
            "out/customer_metrics.csv",
        )
        for expected in facts["rows"]:
            customer_id = expected["canonical_customer_id"]
            actual = by_customer.get(customer_id)
            for field, expected_value in expected.items():
                if field == "canonical_customer_id":
                    continue
                add(
                    "customer_metrics",
                    f"{customer_id}_{field}",
                    "exact",
                    expected_value,
                    _field(actual, field),
                    "out/customer_metrics.csv",
                )
        summary = observed("out/region_summary.json")
        for region, expected in sorted(facts["region_summary_expected"].items()):
            for field, value in expected.items():
                add(
                    "region_summary",
                    f"{region}_{field}",
                    "exact",
                    value,
                    _field(_field(summary, region), field),
                    "out/region_summary.json",
                )
        audit = observed("out/reconciliation_audit.json")
        for field, value in sorted(facts["audit_expected"].items()):
            add(
                "reconciliation_audit",
                field,
                "exact",
                value,
                _field(audit, field),
                "out/reconciliation_audit.json",
            )
        for index, term in enumerate(facts.get("required_notes_terms", []), 1):
            add(
                "reconciliation_notes",
                f"topic_{index:02d}",
                "semantic",
                term,
                observed("out/reconciliation_notes.md"),
                "out/reconciliation_notes.md",
            )
        original = {
            path.relative_to(source / "fixtures").as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in (source / "fixtures/in").rglob("*")
            if path.is_file()
        }
        submitted = {
            path: sha256(raw).hexdigest() for path, raw in files.items() if path.startswith("in/")
        }
        add("input_preserved", "fixture_hashes", "exact", original, submitted, "fixture_hashes")

    if not checks or len({item["id"] for item in checks}) != len(checks):
        raise ComparisonPacketError("Comparison requirements are empty or duplicated")
    for criterion in criteria:
        if not any(check["criterion"] == criterion["id"] for check in checks):
            raise ComparisonPacketError(f"Criterion has no comparison checks: {criterion['id']}")
    artifacts["H01"] = {
        "path": "fixture_hashes",
        "present": True,
        "content": "Only SHA-256 digests are shown",
    }
    packet = {
        "schema_version": SCHEMA_VERSION,
        "canonical_question": question,
        "reference_requirements": [
            {
                key: value
                for key, value in check.items()
                if key not in {"observed", "submission_evidence_id"}
            }
            for check in checks
        ],
        "submitted_answer": {
            "artifacts": artifacts,
            "observations": [
                {
                    "id": check["id"],
                    "observed": check["observed"],
                    "evidence_id": check["submission_evidence_id"],
                    "exact_match": not impossible_credit(check, check["observed"], 4)
                    if check["kind"] == "exact"
                    else None,
                }
                for check in checks
            ],
        },
        "criterion_weights": {item["id"]: float(item["weight"]) for item in criteria},
    }
    if (
        len(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        > MAX_PACKET_BYTES
    ):
        raise ComparisonPacketError("Comparison packet exceeds 400000 bytes; it was not truncated")
    return packet


def packet_from_archive(
    cell: SourceCell, task_root: Path, rubric: dict[str, Any], workspace_image: str
) -> tuple[dict[str, Any], str, str]:
    if cell.archive_path is None:
        raise ComparisonPacketError("The saved cell has no workspace archive")
    files, archive_hash, workspace_hash = read_workspace_archive(cell.archive_path)
    if cell.workspace_hash and workspace_hash != cell.workspace_hash:
        raise ComparisonPacketError("Archived workspace hash differs from v14")
    validation = (
        run_code_test_evidence(files, workspace_image)
        if cell.task_id == "016-code-repair-pytest"
        else None
    )
    packet = build_comparison_packet(
        cell.task_id, task_root / cell.task_id / "source", files, rubric, validation
    )
    return packet, archive_hash, workspace_hash
