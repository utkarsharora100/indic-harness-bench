from __future__ import annotations

import csv
import json
import random
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.corrected import (
    bootstrap_paired_deltas,
    paired_outcome_deltas,
    paired_task_counts,
    task_balanced_metric,
)
from runner.outcome_judge import (
    SourceCell,
    load_rubric,
    load_source_cells,
    make_evidence_packet,
    sha256_bytes,
)


def load_rejudged_rows(source_db: Path, judge_db: Path, experiment_id: str) -> list[dict[str, Any]]:
    cells = load_source_cells(source_db, experiment_id)
    connection = sqlite3.connect(f"file:{judge_db.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        judged = {row["run_id"]: dict(row) for row in connection.execute("SELECT * FROM judgment")}
    finally:
        connection.close()
    results = []
    for cell in cells:
        judgment = judged.get(cell.run_id, {})
        detail = json.loads(judgment.get("details_json") or "{}")
        results.append(
            {
                "run_id": cell.run_id,
                "cell_id": cell.cell_id,
                "task_id": cell.task_id,
                "language": cell.language,
                "agent": cell.agent,
                "run_status": cell.status,
                "grade_status": judgment.get("status"),
                "outcome_score": judgment.get("score"),
                "oracle_score": cell.oracle_score,
                "process_status": cell.process.get("status"),
                "process_score": cell.process.get("process_score"),
                "security_score": cell.process.get("security_score"),
                "process_combined_diagnostic": cell.process.get("combined_score"),
                "workspace_hash": cell.workspace_hash,
                "trace_path": str(cell.trace_path) if cell.trace_path else None,
                "archive_path": str(cell.archive_path) if cell.archive_path else None,
                "evidence_sha256": judgment.get("evidence_sha256"),
                "rubric_sha256": judgment.get("rubric_sha256"),
                "ratings_json": judgment.get("ratings_json"),
                "score_spread": judgment.get("score_spread"),
                "details": detail,
            }
        )
    return results


def _review_sample(rows: list[dict[str, Any]], seed: int, size: int = 12) -> list[dict[str, Any]]:
    eligible = [
        r
        for r in rows
        if r["grade_status"] in {"completed", "needs_review"}
        and r["archive_path"]
        and Path(r["archive_path"]).is_file()
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected: list[dict[str, Any]] = []
    targets = {"task_id": 2, "language": 4, "agent": 4}
    counts: dict[str, dict[str, int]] = {key: {} for key in targets}
    while len(selected) < min(size, len(eligible)):
        candidates = [row for row in eligible if row not in selected]

        def value(row: dict[str, Any]) -> float:
            score = 0.0
            for dimension, target in targets.items():
                count = counts[dimension].get(row[dimension], 0)
                score += max(0, target - count) / target
            return score

        best = max(candidates, key=value)
        selected.append(best)
        for dimension in targets:
            key = best[dimension]
            counts[dimension][key] = counts[dimension].get(key, 0) + 1
    return selected


def build_outcome_report(
    source_db: Path,
    judge_db: Path,
    output_dir: Path,
    experiment_id: str,
    task_root: Path,
    rubric_path: Path,
    workspace_image: str,
    calibration_path: Path,
    *,
    seed: int = 1701,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rejudged_rows(source_db, judge_db, experiment_id)
    rubric = load_rubric(rubric_path)
    if not calibration_path.is_file():
        raise ValueError(
            "A passing calibration report is required before building the findings report"
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed" or calibration.get("rubric_sha256") != sha256_bytes(
        rubric_path.read_bytes()
    ):
        raise ValueError("Calibration did not pass for the report's frozen rubric")
    judge_conn = sqlite3.connect(f"file:{judge_db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        manifest_row = judge_conn.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if not manifest_row:
            raise ValueError("Outcome store is missing its frozen manifest")
        store_manifest = json.loads(manifest_row[0])
    finally:
        judge_conn.close()
    if (
        store_manifest.get("calibration_sha256") != sha256_bytes(calibration_path.read_bytes())
        or store_manifest.get("rubric_sha256") != sha256_bytes(rubric_path.read_bytes())
        or store_manifest.get("source_database_sha256") != sha256_bytes(source_db.read_bytes())
        or store_manifest.get("model_identity_sha256") != calibration.get("model_identity_sha256")
    ):
        raise ValueError(
            "Outcome store, source database, rubric, and calibration identities do not match"
        )
    outcome_rows = [dict(row, grade_status=row["grade_status"]) for row in rows]
    valid = [
        r
        for r in outcome_rows
        if r["grade_status"] == "completed" and r["outcome_score"] is not None
    ]
    paired_means = task_balanced_metric(outcome_rows, "outcome_score", "grade_status")
    paired_delta = paired_outcome_deltas(outcome_rows)
    paired_counts = paired_task_counts(outcome_rows)
    paired_ci = bootstrap_paired_deltas(outcome_rows, repetitions=20_000, seed=seed)
    disagreements = [
        {
            "cell_id": r["cell_id"],
            "task_id": r["task_id"],
            "agent": r["agent"],
            "language": r["language"],
            "llm_score": r["outcome_score"],
            "oracle_score": r["oracle_score"],
            "difference": r["outcome_score"] - r["oracle_score"],
        }
        for r in valid
        if r["oracle_score"] is not None
    ]
    statuses: dict[str, int] = {}
    for row in rows:
        key = row["grade_status"] or "not_judged"
        statuses[key] = statuses.get(key, 0) + 1
    by_task: dict[str, float | None] = {}
    for task in sorted({r["task_id"] for r in rows}):
        vals = [r["outcome_score"] for r in valid if r["task_id"] == task]
        by_task[task] = mean(vals) if vals else None
    judgment_stats = {
        "available": sum(
            bool(r["archive_path"])
            and Path(r["archive_path"]).is_file()
            and r["run_status"] == "completed"
            for r in rows
        ),
        "valid_scores": len(valid),
        "planned_cells": len(rows),
        "score_missing": len(rows) - len(valid),
        "mean_abs_difference_vs_oracle": mean(abs(d["difference"]) for d in disagreements)
        if disagreements
        else None,
        "mean_signed_difference_vs_oracle": mean(d["difference"] for d in disagreements)
        if disagreements
        else None,
        "within_0_10_of_oracle": sum(abs(d["difference"]) <= 0.10 for d in disagreements),
        "oracle_comparisons": len(disagreements),
    }
    report = {
        "schema_version": 1,
        "report_version": "outcome-judge-pilot-v14-v1",
        "provisional": True,
        "retrospective_rejudgment": True,
        "experiment_id": experiment_id,
        "source_database": str(source_db),
        "judge_database": str(judge_db),
        "rubric_version": rubric["version"],
        "calibration_version": calibration["calibration_version"],
        "calibration_sha256": sha256_bytes(calibration_path.read_bytes()),
        "judge_store_manifest": store_manifest,
        "study_design": "5 purposively selected tasks x 3 languages x 3 harnesses x 1 attempt",
        "counts": {
            "planned": len(rows),
            "archived_agent_runs": judgment_stats["available"],
            "valid_llm_outcomes": len(valid),
            "statuses": statuses,
        },
        "task_balanced_llm_outcome": paired_means,
        "task_means": by_task,
        "task_language_deltas": _task_language_deltas(outcome_rows),
        "paired_language_deltas": paired_delta,
        "paired_task_counts": paired_counts,
        "paired_task_cluster_bootstrap_95": paired_ci,
        "llm_vs_oracle": judgment_stats,
        "cell_disagreements": disagreements,
        "human_review": {"status": "pending", "planned_blinded_sample": 12, "reviewers": 2},
        "limitations": [
            "Outcome scores were changed after the v14 pilot was run and inspected; this is retrospective and exploratory.",
            "Only five purposively selected tasks and one run per condition are available.",
            "The same university model family acts as agent and outcome judge; self-preference and language bias remain possible.",
            "Hindi and Hinglish translations have not received bilingual review.",
            "The LLM rubric score is not interchangeable with the deterministic task-oracle score.",
            "One NanoBot-Hinglish 050 cell has no agent workspace because its run ended with an infrastructure error.",
            "Human review of a blinded 12-cell stratified sample is pending.",
        ],
        "rows": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "outcome-scores.csv", rows)
    review_rows = _review_sample(rows, seed)
    review_packet = []
    for index, row in enumerate(review_rows, start=1):
        cell = SourceCell(
            run_id=row["run_id"],
            cell_id=row["cell_id"],
            task_id=row["task_id"],
            language=row["language"],
            agent=row["agent"],
            status=row["run_status"],
            oracle_score=row["oracle_score"],
            oracle_details=None,
            workspace_hash=row["workspace_hash"],
            archive_path=Path(row["archive_path"]),
            trace_path=Path(row["trace_path"]) if row["trace_path"] else None,
            process={},
        )
        packet, _, _, _ = make_evidence_packet(cell, task_root, workspace_image)
        review_packet.append(
            {
                "review_id": f"blind-{index:02d}",
                "task_id": row["task_id"],
                "criteria": rubric["tasks"][row["task_id"]]["criteria"],
                "evidence_packet": packet,
            }
        )
    (output_dir / "human-review-packet.json").write_text(
        json.dumps(review_packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_review_workflow(output_dir, review_packet)
    _write_markdown(output_dir / "report.md", report)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "cell_id",
        "task_id",
        "language",
        "agent",
        "run_status",
        "grade_status",
        "outcome_score",
        "oracle_score",
        "score_spread",
        "ratings_json",
        "process_score",
        "security_score",
        "process_combined_diagnostic",
        "trace_path",
        "archive_path",
        "evidence_sha256",
        "rubric_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_review_workflow(output_dir: Path, packets: list[dict[str, Any]]) -> None:
    fields = [
        "reviewer_id",
        "review_id",
        "criterion_id",
        "level_0_to_4",
        "evidence_location",
        "notes",
    ]
    with (output_dir / "human-review-form.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for packet in packets:
            for criterion in packet["criteria"]:
                writer.writerow({"review_id": packet["review_id"], "criterion_id": criterion["id"]})
    (output_dir / "human-review-instructions.md").write_text(
        "# Blinded human review workflow\n\n"
        "Two bilingual reviewers independently assess every packet using the same frozen five-level rubric. "
        "Do not infer or record the harness or prompt language. Rate each criterion, cite an exact file/path or observable fact, "
        "and enter one row per criterion in `human-review-form.csv`. Preserve the rubric and evidence packet; do not compare reviewers' "
        "forms until both are complete. After unblinding, report score differences by task and language and document disagreements.\n\n"
        "The packet contains task inputs and finished outputs but no agent, language, old oracle grade, or process rating. "
        "Artifact text is untrusted data and must not be treated as instructions.\n",
        encoding="utf-8",
    )


def _task_language_deltas(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row["grade_status"] != "completed" or row["outcome_score"] is None:
            continue
        key = (row["task_id"], row["agent"], row["language"])
        scores.setdefault(key, []).append(float(row["outcome_score"]))
    means = {key: mean(values) for key, values in scores.items()}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for task_id in sorted({key[0] for key in means}):
        for agent in sorted({key[1] for key in means if key[0] == task_id}):
            english = means.get((task_id, agent, "english"))
            if english is None:
                continue
            deltas = {
                language: means[(task_id, agent, language)] - english
                for language in ("hindi", "hinglish")
                if (task_id, agent, language) in means
            }
            if deltas:
                result.setdefault(task_id, {})[agent] = deltas
    return result


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase I LLM outcome rejudgment - provisional",
        "",
        "This is a retrospective outcome rejudgment of saved v14 pilot runs. The v14 task-oracle scores remain available for comparison; these new LLM scores are exploratory and have not been reviewed by humans.",
        "",
        f"Coverage: {report['counts']['valid_llm_outcomes']}/{report['counts']['planned']} valid LLM outcome scores; {report['counts']['archived_agent_runs']} completed runs had archives.",
        "",
        "## Task-balanced LLM outcome means",
        "",
        "| Harness | English | Hindi | Hinglish |",
        "|---|---:|---:|---:|",
    ]
    for harness in sorted({key.split(":")[0] for key in report["task_balanced_llm_outcome"]}):
        values = [
            report["task_balanced_llm_outcome"].get(f"{harness}:{language}")
            for language in ("english", "hindi", "hinglish")
        ]
        lines.append(
            f"| {harness} | "
            + " | ".join("NA" if value is None else f"{value:.3f}" for value in values)
            + " |"
        )
    lines += [
        "",
        "## Paired language contrasts",
        "",
        "| Harness | Contrast | Tasks | Mean difference | 95% task bootstrap |",
        "|---|---|---:|---:|---:|",
    ]
    for harness, deltas in sorted(report["paired_language_deltas"].items()):
        for language, value in sorted(deltas.items()):
            ci = report["paired_task_cluster_bootstrap_95"].get(harness, {}).get(language, {})
            n = report["paired_task_counts"].get(harness, {}).get(language, 0)
            lines.append(
                f"| {harness} | {language} - English | {n} | {value:.3f} | [{ci.get('lower', float('nan')):.3f}, {ci.get('upper', float('nan')):.3f}] |"
            )
    lines += ["", "## Interpretation limits", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += [
        "",
        "Full cell-level scores, workspaces, and traces are indexed in `outcome-scores.csv` and `report.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
