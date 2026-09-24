from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.corrected import (
    bootstrap_paired_deltas,
    paired_outcome_deltas,
    paired_task_counts,
    task_balanced_metric,
)
from runner.outcome_judge import load_rubric, load_source_cells, sha256_bytes, validate_judgment
from runner.outcome_v2 import (
    EXPECTED_SOURCE_DB_SHA256,
    HARNESSES,
    KNOWN_MISSING,
    LANGUAGES,
    MAX_CALL_ATTEMPTS,
    OUTCOME_VERSION,
    ROUND_ID,
    TASK_IDS,
    OutcomeV2Error,
    _code_manifest,
    _load_passes,
    _pass_requires_third,
    _scan_for_secrets,
    verify_frozen_inputs,
)
from runner.outcome_contract import model_response_from_ratings


def _ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise OutcomeV2Error(f"Required rejudgment store is missing: {path.name}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _column_usage(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row.get(field) for row in rows]
    observed = [int(value) for value in values if isinstance(value, int) and value >= 0]
    return {
        "sum_observed": sum(observed) if observed else None,
        "calls_with_usage": len(observed),
        "calls_missing_usage": len(values) - len(observed),
    }


def validate_judgment_store(
    *,
    root: Path,
    source_database: Path,
    judge_database: Path,
    calibration_path: Path,
    rubric_path: Path,
    experiment_id: str,
    secrets: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verify_frozen_inputs(root, source_database)
    rubric = load_rubric(rubric_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rubric_hash = sha256_bytes(rubric_path.read_bytes())
    if (
        calibration.get("status") not in {"passed", "failed_substantive_controls"}
        or calibration.get("outcome_version") != OUTCOME_VERSION
        or calibration.get("rubric_sha256") != rubric_hash
        or calibration.get("source_database_sha256") != EXPECTED_SOURCE_DB_SHA256
        or calibration.get("evidence_schema_version") != 5
        or not calibration.get("transport_smoke", {}).get("status") == "passed"
        or len(calibration.get("results", {})) != 29
        or len(calibration.get("checks", [])) != 15
        or len(calibration.get("deterministic_oracle_checks", [])) != 20
        or (calibration["status"] == "passed")
        != all(item.get("passed") for item in calibration["checks"])
        or not all(
            item.get("passed") for item in calibration.get("deterministic_oracle_checks", [])
        )
    ):
        raise OutcomeV2Error("Calibration is incomplete, failed, or has mismatched identities")

    connection = _ro(judge_database)
    try:
        manifest_row = connection.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if manifest_row is None:
            raise OutcomeV2Error("Judgment store has no frozen manifest")
        manifest = json.loads(manifest_row[0])
        if (
            manifest.get("outcome_version") != OUTCOME_VERSION
            or manifest.get("source_database_sha256") != sha256_bytes(source_database.read_bytes())
            or manifest.get("rubric_sha256") != rubric_hash
            or manifest.get("calibration_sha256") != sha256_bytes(calibration_path.read_bytes())
            or manifest.get("model_identity_sha256") != calibration.get("model_identity_sha256")
            or manifest.get("source_task_hashes") != calibration.get("source_task_hashes")
            or manifest.get("code_sha256") != _code_manifest(root)
        ):
            raise OutcomeV2Error(
                "Judgment store, calibration, model, rubric, or code versions differ"
            )

        source_cells = load_source_cells(source_database, experiment_id)
        if len(source_cells) != 45:
            raise OutcomeV2Error("The frozen v14 study must contain 45 source cells")
        rows = [dict(row) for row in connection.execute("SELECT * FROM subject_cell")]
        if len(rows) != 45 or {row["run_id"] for row in rows} != {
            cell.run_id for cell in source_cells
        }:
            raise OutcomeV2Error("Judgment store does not index all 45 v14 source cells")

        source_by_id = {cell.run_id: cell for cell in source_cells}
        if len({(row["task_id"], row["language"], row["harness"]) for row in rows}) != 45:
            raise OutcomeV2Error("Judgment store has duplicate or missing matrix conditions")
        status_counts = Counter(row["status"] for row in rows)
        if (
            status_counts.get("missing_agent_artifact") != 1
            or status_counts.get("pending", 0)
            or status_counts.get("invalid_evidence", 0)
            or status_counts.get("completed", 0)
            + status_counts.get("needs_review", 0)
            + status_counts.get("judge_error", 0)
            != 44
        ):
            raise OutcomeV2Error(
                "Report blocked: expected 44 judged or explicit judge-error cells and one known missing artifact"
            )
        missing = [row for row in rows if row["status"] == "missing_agent_artifact"]
        if (
            len(missing) != 1
            or (missing[0]["task_id"], missing[0]["language"], missing[0]["harness"])
            != KNOWN_MISSING
        ):
            raise OutcomeV2Error(
                "The missing cell is not the known v14 NanoBot/Hinglish/task-050 run"
            )

        for row in rows:
            source = source_by_id[row["run_id"]]
            if (
                row["cell_id"] != source.cell_id
                or row["task_id"] != source.task_id
                or row["language"] != source.language
                or row["harness"] != source.agent
                or row["run_status"] != source.status
                or not row["trace_path"]
                or not Path(row["trace_path"]).is_file()
                or sha256_bytes(Path(row["trace_path"]).read_bytes()) != row["trace_sha256"]
            ):
                raise OutcomeV2Error(
                    "A source trace or source-cell identity changed after preflight"
                )
            if row["status"] == "missing_agent_artifact":
                continue
            evidence = connection.execute(
                "SELECT * FROM evidence_packet WHERE run_id=?", (row["run_id"],)
            ).fetchone()
            if evidence is None:
                raise OutcomeV2Error("A completed source cell has no frozen evidence packet")
            packet = json.loads(evidence["packet_json"])
            if (
                sha256_bytes(evidence["packet_json"].encode("utf-8")) != evidence["packet_sha256"]
                or evidence["packet_sha256"] != row["evidence_sha256"]
                or evidence["rubric_sha256"] != rubric_hash
                or evidence["trace_sha256"] != row["trace_sha256"]
                or evidence["archive_sha256"] != row["archive_sha256"]
                or evidence["workspace_sha256"] != row["workspace_hash"]
            ):
                raise OutcomeV2Error("Frozen evidence packet hashes do not validate")
            criteria = rubric["tasks"][row["task_id"]]["criteria"]
            passes = _load_passes(connection, row["run_id"], ROUND_ID)
            if row["status"] == "judge_error":
                if row["score"] is not None:
                    raise OutcomeV2Error("Judge-error cell must not have an outcome score")
                detail = json.loads(row["details_json"] or "{}")
                if not detail.get("judge_error"):
                    raise OutcomeV2Error("Judge-error cell lacks a recorded failure reason")
                continue
            if len(passes) not in {2, 3} or set(passes) != set(range(1, len(passes) + 1)):
                raise OutcomeV2Error(
                    "Every judged cell must have two or three complete independent passes"
                )
            if len(passes) == 2 and _pass_requires_third(passes[1], passes[2]):
                raise OutcomeV2Error("A cell requiring a third pass is missing it")
            if len(passes) == 3 and row["status"] not in {"completed", "needs_review"}:
                raise OutcomeV2Error("Third-pass judgment has an invalid final status")
            if (row["status"] == "needs_review") != (
                len(passes) == 3 and float(row["score_spread"]) > 0.20
            ):
                raise OutcomeV2Error(
                    "Review-needed status does not match the frozen disagreement rule"
                )
            for pass_no, judged in passes.items():
                validate_judgment(
                    json.dumps(
                        model_response_from_ratings(judged["ratings"]),
                        ensure_ascii=False,
                    ),
                    criteria,
                    packet,
                )
                attempt_rows = connection.execute(
                    "SELECT * FROM call_attempt WHERE subject_id=? AND round_id=? AND pass_no=? ORDER BY attempt_no",
                    (row["run_id"], ROUND_ID, pass_no),
                ).fetchall()
                if not attempt_rows or len(attempt_rows) > MAX_CALL_ATTEMPTS:
                    raise OutcomeV2Error(
                        "Call-attempt history is missing or exceeds its retry limit"
                    )
                deterministic_only = (
                    len(attempt_rows) == 1 and attempt_rows[0]["status"] == "deterministic"
                )
                if not deterministic_only and (
                    sum(item["status"] == "completed" for item in attempt_rows) != 1
                    or attempt_rows[-1]["status"] != "completed"
                ):
                    raise OutcomeV2Error(
                        "A successful pass lacks one final successful call attempt"
                    )
                if any(
                    item["status"] not in {"completed", "deterministic", "error"}
                    for item in attempt_rows
                ):
                    raise OutcomeV2Error("Call-attempt history contains an invalid status")
                for item in attempt_rows:
                    for token_field in ("input_tokens", "output_tokens", "total_tokens"):
                        value = item[token_field]
                        if value is not None and (not isinstance(value, int) or value < 0):
                            raise OutcomeV2Error("Call usage contains an invalid token count")
                    if item["raw_response"]:
                        _scan_for_secrets_from_bytes(item["raw_response"].encode("utf-8"), secrets)

        stray = connection.execute(
            "SELECT COUNT(*) FROM call_attempt WHERE subject_id NOT IN (SELECT run_id FROM subject_cell)"
        ).fetchone()[0]
        if stray:
            raise OutcomeV2Error(
                "Judgment store contains attempts not linked to planned source cells"
            )
        for attempt in connection.execute("SELECT * FROM call_attempt"):
            for token_field in ("input_tokens", "output_tokens", "total_tokens"):
                value = attempt[token_field]
                if value is not None and (not isinstance(value, int) or value < 0):
                    raise OutcomeV2Error("Call usage contains an invalid token count")
            for field in ("raw_response", "error"):
                if attempt[field]:
                    _scan_for_secrets_from_bytes(attempt[field].encode("utf-8"), secrets)
        proxy_rows = connection.execute("SELECT payload_json FROM proxy_event").fetchall()
        for proxy_row in proxy_rows:
            _scan_for_secrets_from_bytes(proxy_row[0].encode("utf-8"), secrets)
        return rows, manifest, calibration
    finally:
        connection.close()


def _scan_for_secrets_from_bytes(raw: bytes, secrets: tuple[str, ...]) -> None:
    if any(value.encode("utf-8") in raw for value in secrets if value):
        raise OutcomeV2Error("Private inference value found in a saved judge record")


def _review_strata(rows: list[dict[str, Any]]) -> dict[str, str]:
    eligible: list[tuple[float, str]] = []
    for row in rows:
        if row["status"] not in {"completed", "needs_review"}:
            continue
        if row["score"] is not None:
            llm_score = float(row["score"])
        else:
            scores = json.loads(row["pass_scores_json"] or "[]")
            if not scores:
                continue
            llm_score = mean(float(value) for value in scores)
        if row["oracle_score"] is None:
            continue
        eligible.append((abs(llm_score - float(row["oracle_score"])), row["run_id"]))
    eligible.sort()
    result = {}
    for index, (_difference, run_id) in enumerate(eligible):
        band = min(2, index * 3 // max(1, len(eligible)))
        result[run_id] = ("low", "middle", "high")[band]
    return result


def select_review_sample(rows: list[dict[str, Any]], seed: int = 1701) -> list[dict[str, Any]]:
    strata = _review_strata(rows)
    candidates = [
        row
        for row in rows
        if row["run_id"] in strata and row["status"] in {"completed", "needs_review"}
    ]
    if len(candidates) < 12:
        raise OutcomeV2Error(
            "At least 12 reviewed or review-needed cells are required for the human sample"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Stable shuffled priority gives a reproducible exact quota search.
    candidates.sort(
        key=lambda row: (
            row["task_id"],
            row["language"],
            row["harness"],
            strata[row["run_id"]],
        )
    )
    tie_order = {
        row["run_id"]: index for index, row in enumerate(rng.sample(candidates, len(candidates)))
    }
    candidates.sort(key=lambda row: tie_order[row["run_id"]])

    selected: list[dict[str, Any]] = []
    lang_counts: Counter[str] = Counter()
    harness_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    nodes = 0

    def feasible(start: int) -> bool:
        remaining = candidates[start:]
        if any(lang_counts[lang] > 4 for lang in LANGUAGES):
            return False
        if any(harness_counts[harness] > 4 for harness in HARNESSES):
            return False
        for lang in LANGUAGES:
            if lang_counts[lang] + sum(row["language"] == lang for row in remaining) < 4:
                return False
        for harness in HARNESSES:
            if harness_counts[harness] + sum(row["harness"] == harness for row in remaining) < 4:
                return False
        for task in TASK_IDS:
            if task_counts[task] + sum(row["task_id"] == task for row in remaining) < 2:
                return False
        for band in ("low", "middle", "high"):
            if band_counts[band] == 0 and not any(
                strata[row["run_id"]] == band for row in remaining
            ):
                return False
        return True

    def search(start: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > 2_000_000 or len(selected) > 12 or not feasible(start):
            return False
        if len(selected) == 12:
            return (
                all(lang_counts[lang] == 4 for lang in LANGUAGES)
                and all(harness_counts[harness] == 4 for harness in HARNESSES)
                and all(task_counts[task] >= 2 for task in TASK_IDS)
                and all(band_counts[band] >= 1 for band in ("low", "middle", "high"))
            )
        remaining_needed = 12 - len(selected)
        if len(candidates) - start < remaining_needed:
            return False
        for index in range(start, len(candidates)):
            row = candidates[index]
            if lang_counts[row["language"]] >= 4 or harness_counts[row["harness"]] >= 4:
                continue
            selected.append(row)
            lang_counts[row["language"]] += 1
            harness_counts[row["harness"]] += 1
            task_counts[row["task_id"]] += 1
            band_counts[strata[row["run_id"]]] += 1
            if search(index + 1):
                return True
            selected.pop()
            lang_counts[row["language"]] -= 1
            harness_counts[row["harness"]] -= 1
            task_counts[row["task_id"]] -= 1
            band_counts[strata[row["run_id"]]] -= 1
        return False

    if not search(0):
        raise OutcomeV2Error("Cannot satisfy deterministic 12-cell human-review quotas")
    return [dict(row, disagreement_band=strata[row["run_id"]]) for row in selected]


def _score_rows(
    source_database: Path,
    rows: list[dict[str, Any]],
    judge_database: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _ro(source_database)
    judge = _ro(judge_database)
    try:
        usage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for attempt in judge.execute(
            "SELECT * FROM call_attempt ORDER BY subject_id,pass_no,attempt_no"
        ):
            usage[attempt["subject_id"]].append(dict(attempt))
        task_names = {
            row["task_id"]: row["task_id"] for row in source.execute("SELECT task_id FROM task")
        }
        score_rows = []
        criterion_rows = []
        for row in rows:
            attempts = usage.get(row["run_id"], [])
            input_tokens = _column_usage(attempts, "input_tokens")
            output_tokens = _column_usage(attempts, "output_tokens")
            total_tokens = _column_usage(attempts, "total_tokens")
            llm_process_security = None
            if all(
                row.get(field) is not None for field in ("score", "process_score", "security_score")
            ):
                llm_process_security = (
                    float(row["score"]) * float(row["process_score"]) * float(row["security_score"])
                )
            detail = json.loads(row["details_json"] or "{}")
            score_row = {
                "run_id": row["run_id"],
                "cell_id": row["cell_id"],
                "task_id": row["task_id"],
                "task": task_names.get(row["task_id"]),
                "language": row["language"],
                "harness": row["harness"],
                "run_status": row["run_status"],
                "status": row["status"],
                "outcome_score": row["score"],
                "oracle_score": row["oracle_score"],
                "process_score": row["process_score"],
                "security_score": row["security_score"],
                "llm_outcome_process_security_diagnostic": llm_process_security,
                "score_spread": row["score_spread"],
                "pass_scores_json": row["pass_scores_json"],
                "ratings_json": row["ratings_json"],
                "judge_calls": len(attempts),
                "judge_failed_calls": sum(item["status"] == "error" for item in attempts),
                "judge_input_tokens": input_tokens["sum_observed"],
                "judge_input_usage_missing_calls": input_tokens["calls_missing_usage"],
                "judge_output_tokens": output_tokens["sum_observed"],
                "judge_output_usage_missing_calls": output_tokens["calls_missing_usage"],
                "judge_total_tokens": total_tokens["sum_observed"],
                "judge_total_usage_missing_calls": total_tokens["calls_missing_usage"],
                "trace_path": row["trace_path"],
                "trace_sha256": row["trace_sha256"],
                "workspace_archive": row["archive_path"],
                "workspace_archive_sha256": row["archive_sha256"],
                "workspace_sha256": row["workspace_hash"],
                "evidence_sha256": row["evidence_sha256"],
                "detail_json": json.dumps(detail, ensure_ascii=False, sort_keys=True),
            }
            score_rows.append(score_row)
            if row["ratings_json"]:
                ratings = json.loads(row["ratings_json"])
                for criterion_id, rating in ratings.items():
                    criterion_rows.append(
                        {
                            "run_id": row["run_id"],
                            "cell_id": row["cell_id"],
                            "task_id": row["task_id"],
                            "language": row["language"],
                            "harness": row["harness"],
                            "status": row["status"],
                            "criterion_id": criterion_id,
                            "weight": rating["weight"],
                            "median_or_mean_level": rating["level"],
                            "pass_evidence_json": json.dumps(
                                rating["evidence"], ensure_ascii=False
                            ),
                            "reference_fields_json": json.dumps(
                                rating.get("reference_fields", ()), ensure_ascii=False
                            ),
                            "model_levels_json": json.dumps(
                                rating.get("model_levels", ()), ensure_ascii=False
                            ),
                            "rationale_statuses_json": json.dumps(
                                rating.get("rationale_statuses", ()), ensure_ascii=False
                            ),
                            "pass_rationales_json": json.dumps(
                                rating["rationales"], ensure_ascii=False
                            ),
                        }
                    )
        return score_rows, criterion_rows
    finally:
        source.close()
        judge.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_human_review(
    output_dir: Path,
    selected: list[dict[str, Any]],
    rubric: dict[str, Any],
    store: sqlite3.Connection,
) -> None:
    packets = []
    unblinding = []
    forms: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for index, row in enumerate(selected, start=1):
        evidence = store.execute(
            "SELECT packet_json FROM evidence_packet WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        if evidence is None:
            raise OutcomeV2Error("Selected human-review cell lacks frozen evidence")
        review_id = f"blind-{index:02d}"
        criteria = rubric["tasks"][row["task_id"]]["criteria"]
        review_packet = json.loads(evidence["packet_json"])
        for rule in review_packet.get("criterion_contract", {}).values():
            rule.pop("forced_level", None)
            rule.pop("max_level", None)
        packets.append(
            {
                "review_id": review_id,
                "task_id": row["task_id"],
                "criteria": criteria,
                "evidence_packet": review_packet,
            }
        )
        unblinding.append(
            {
                "review_id": review_id,
                "run_id": row["run_id"],
                "cell_id": row["cell_id"],
                "task_id": row["task_id"],
                "language": row["language"],
                "harness": row["harness"],
                "llm_score": row["score"],
                "oracle_score": row["oracle_score"],
                "disagreement_band": row["disagreement_band"],
            }
        )
        for reviewer in forms:
            for criterion in criteria:
                forms[reviewer].append(
                    {
                        "review_id": review_id,
                        "task_id": row["task_id"],
                        "criterion_id": criterion["id"],
                        "criterion_weight": criterion["weight"],
                        "level_0_to_4": "",
                        "submitted_file_or_test": "",
                        "reference_fact_used": "",
                        "rationale": "",
                    }
                )
    (output_dir / "human-review-packet.json").write_text(
        json.dumps(packets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "human-review-key.json").write_text(
        json.dumps(unblinding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for reviewer, values in forms.items():
        _write_csv(output_dir / f"human-reviewer-{reviewer}.csv", values)
    (output_dir / "human-review-instructions.md").write_text(
        "# Blinded bilingual review\n\n"
        "Two bilingual reviewers independently use the same frozen five-level rubric. "
        "Rate each criterion, identify the submitted file or test used, and record one short rationale. "
        "Do not infer or record language, harness, old oracle score, process score, or LLM score. "
        "Return completed reviewer A and reviewer B forms separately; compare only after both are locked.\n",
        encoding="utf-8",
    )


def _task_categories(root: Path, dataset_manifest: str) -> dict[str, str]:
    import yaml

    selection = yaml.safe_load((root / dataset_manifest).read_text(encoding="utf-8")) or {}
    return {item["task_id"]: item["category"] for item in selection.get("tasks", [])}


def build_outcome_report_v2(
    *,
    root: Path,
    source_database: Path,
    judge_database: Path,
    calibration_path: Path,
    rubric_path: Path,
    output_dir: Path,
    experiment_id: str,
    dataset_manifest: str,
    secrets: tuple[str, ...],
    seed: int = 1701,
) -> dict[str, Any]:
    rows, store_manifest, calibration = validate_judgment_store(
        root=root,
        source_database=source_database,
        judge_database=judge_database,
        calibration_path=calibration_path,
        rubric_path=rubric_path,
        experiment_id=experiment_id,
        secrets=secrets,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rubric = load_rubric(rubric_path)
    scores, criterion_rows = _score_rows(source_database, rows, judge_database)
    metric_rows = [
        {
            **row,
            "grade_status": row["status"],
            "agent": row["harness"],
            "score": row["outcome_score"],
        }
        for row in scores
    ]
    balanced = task_balanced_metric(metric_rows, "outcome_score", "grade_status")
    paired = paired_outcome_deltas(metric_rows)
    paired_counts = paired_task_counts(metric_rows)
    bootstrap = bootstrap_paired_deltas(metric_rows, repetitions=20_000, seed=seed)
    category_map = _task_categories(root, dataset_manifest)
    category_means: dict[str, dict[str, float | None]] = {}
    for category in sorted(set(category_map.values())):
        for harness in HARNESSES:
            for language in LANGUAGES:
                values = [
                    row["outcome_score"]
                    for row in scores
                    if row["status"] == "completed"
                    and row["outcome_score"] is not None
                    and category_map.get(row["task_id"]) == category
                    and row["harness"] == harness
                    and row["language"] == language
                ]
                category_means.setdefault(category, {})[f"{harness}:{language}"] = (
                    mean(values) if values else None
                )
    disagreements = [
        {
            "cell_id": row["cell_id"],
            "task_id": row["task_id"],
            "harness": row["harness"],
            "language": row["language"],
            "llm_score": row["outcome_score"],
            "oracle_score": row["oracle_score"],
            "difference": float(row["outcome_score"]) - float(row["oracle_score"]),
        }
        for row in scores
        if row["status"] == "completed"
        and row["outcome_score"] is not None
        and row["oracle_score"] is not None
    ]
    statuses = Counter(row["status"] for row in rows)
    task_means = {
        task_id: {
            f"{harness}:{language}": mean(
                [
                    row["outcome_score"]
                    for row in scores
                    if row["task_id"] == task_id
                    and row["harness"] == harness
                    and row["language"] == language
                    and row["status"] == "completed"
                    and row["outcome_score"] is not None
                ]
            )
            if any(
                row["task_id"] == task_id
                and row["harness"] == harness
                and row["language"] == language
                and row["status"] == "completed"
                and row["outcome_score"] is not None
                for row in scores
            )
            else None
            for harness in HARNESSES
            for language in LANGUAGES
        }
        for task_id in TASK_IDS
    }
    by_status_rows = [dict(row) for row in rows]
    try:
        review_sample = select_review_sample(by_status_rows, seed=seed)
    except OutcomeV2Error:
        eligible = [row for row in by_status_rows if row["status"] in {"completed", "needs_review"}]
        random.Random(seed).shuffle(eligible)
        review_sample = [
            dict(row, disagreement_band="unbalanced_fallback") for row in eligible[:12]
        ]
    report = {
        "schema_version": 2,
        "report_version": f"{OUTCOME_VERSION}-pilot-v14-provisional",
        "outcome_version": OUTCOME_VERSION,
        "provisional": True,
        "retrospective_rejudgment": True,
        "experiment_id": experiment_id,
        "study_design": "5 purposively selected tasks x 3 languages x 3 harnesses x 1 execution",
        "primary_outcome": "LLM-rubric weighted completion score in [0,1]",
        "diagnostic_aggregate": "LLM outcome x process score x security score; missing components remain missing",
        "counts": {
            "planned_cells": 45,
            "archived_agent_runs": 44,
            "completed_outcome_scores": sum(row["status"] == "completed" for row in rows),
            "needs_review": sum(row["status"] == "needs_review" for row in rows),
            "judge_error": sum(row["status"] == "judge_error" for row in rows),
            "missing_agent_artifact": 1,
            "judgment_statuses": dict(statuses),
            "paired_task_counts": paired_counts,
        },
        "identities": {
            "source_database_sha256": store_manifest["source_database_sha256"],
            "original_pdf_sha256": store_manifest["original_pdf_sha256"],
            "rubric_sha256": store_manifest["rubric_sha256"],
            "calibration_sha256": store_manifest["calibration_sha256"],
            "model_identity_sha256": store_manifest["model_identity_sha256"],
            "model_public_label": store_manifest["model_public_label"],
            "code_sha256": store_manifest["code_sha256"],
        },
        "calibration": {
            "version": calibration["calibration_version"],
            "status": calibration["status"],
            "control_count": calibration["control_count"],
            "checks": calibration["checks"],
            "checks_passed": sum(bool(item.get("passed")) for item in calibration["checks"]),
            "checks_failed": sum(not item.get("passed") for item in calibration["checks"]),
            "deterministic_oracle_checks": calibration["deterministic_oracle_checks"],
            "transport_smoke": calibration["transport_smoke"],
        },
        "task_balanced_harness_language_means": balanced,
        "task_means": task_means,
        "task_categories": category_map,
        "category_means": category_means,
        "paired_language_deltas": paired,
        "paired_task_counts": paired_counts,
        "paired_task_cluster_bootstrap_95": bootstrap,
        "llm_vs_oracle": {
            "comparisons": len(disagreements),
            "mean_signed_difference": mean(item["difference"] for item in disagreements)
            if disagreements
            else None,
            "mean_absolute_difference": mean(abs(item["difference"]) for item in disagreements)
            if disagreements
            else None,
            "cell_disagreements": disagreements,
        },
        "human_review": {
            "status": "pending",
            "planned_cells": 12,
            "selected_cells": len(review_sample),
            "exact_stratification_met": len(review_sample) == 12
            and all(
                Counter(row["language"] for row in review_sample)[language] == 4
                for language in LANGUAGES
            )
            and all(
                Counter(row["harness"] for row in review_sample)[harness] == 4
                for harness in HARNESSES
            )
            and all(
                Counter(row["task_id"] for row in review_sample)[task] >= 2 for task in TASK_IDS
            ),
            "reviewers": 2,
            "sample_counts": {
                "language": dict(Counter(row["language"] for row in review_sample)),
                "harness": dict(Counter(row["harness"] for row in review_sample)),
                "task": dict(Counter(row["task_id"] for row in review_sample)),
                "disagreement_band": dict(
                    Counter(row["disagreement_band"] for row in review_sample)
                ),
            },
        },
        "limitations": [
            "A failed substantive calibration control means this judge is unvalidated; displayed contrasts are descriptive, not language-performance claims.",
            "This is a retrospective change to the pilot outcome and is exploratory.",
            "The five tasks were purposively selected; one execution per condition is available.",
            "The same pinned university model family generated agent work and judged outcomes.",
            "Hindi and Hinglish translations have not been independently reviewed by bilingual people.",
            "The model judge is not independent human ground truth; a blinded two-reviewer sample is pending.",
            "The NanoBot/Hinglish task-050 cell has no archived agent workspace and remains missing.",
            "The task-001 upstream oracle is binary and does not score input preservation; this limitation is explicitly covered by synthetic controls.",
            "This small corrected pilot is not numerically comparable to the original Harness-Bench leaderboard.",
        ],
        "rows": scores,
        "criterion_rows": criterion_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "outcome-scores.csv", scores)
    _write_csv(output_dir / "criterion-scores.csv", criterion_rows)
    connection = _ro(judge_database)
    try:
        _write_human_review(output_dir, review_sample, rubric, connection)
    finally:
        connection.close()
    _write_markdown(output_dir / "report.md", report)
    _scan_for_secrets([path for path in output_dir.rglob("*") if path.is_file()], secrets)
    verify_frozen_inputs(root, source_database)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Phase I {OUTCOME_VERSION} rejudgment: provisional findings",
        "",
        "This is a retrospective LLM outcome rejudgment of the saved v14 pilot. It covers five purposively selected tasks and one execution per condition. The same pinned university model generated the work and judged it. Bilingual review has not yet been completed.",
        "",
        f"Coverage: {report['counts']['completed_outcome_scores']} numeric outcomes, {report['counts']['needs_review']} review-needed cells, {report['counts']['judge_error']} judge errors, 1 known missing agent artifact, 45 planned source cells.",
        f"Calibration: {report['calibration']['status']} ({report['calibration']['checks_failed']} failed substantive checks). Results are descriptive and not validated language effects if any control failed.",
        "",
        "## Task-balanced means",
        "",
        "| Harness | English | Hindi | Hinglish |",
        "|---|---:|---:|---:|",
    ]
    for harness in HARNESSES:
        values = [
            report["task_balanced_harness_language_means"].get(f"{harness}:{language}")
            for language in LANGUAGES
        ]
        lines.append(
            f"| {harness} | "
            + " | ".join("NA" if value is None else f"{value:.3f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired language contrasts",
            "",
            "| Harness | Contrast | Tasks | Mean difference | 95% task-cluster interval |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for harness, contrasts in report["paired_language_deltas"].items():
        for language, difference in contrasts.items():
            ci = report["paired_task_cluster_bootstrap_95"].get(harness, {}).get(language, {})
            n = report["paired_task_counts"].get(harness, {}).get(language, 0)
            lines.append(
                f"| {harness} | {language} - English | {n} | {difference:.3f} | "
                f"[{ci.get('lower', float('nan')):.3f}, {ci.get('upper', float('nan')):.3f}] |"
            )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
