from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.corrected import (
    bootstrap_paired_deltas,
    error_conditioned_recovery,
    harness_interactions,
    load_corrected_rows,
    paired_outcome_categories,
    paired_outcome_deltas,
    score_difference_distribution,
    task_balanced_outcome,
    task_balanced_metric,
    task_condition_scores,
    trace_links,
    usage_summary,
)


def build_corrected_report(database: Path, output: Path, experiment_id: str) -> dict[str, Any]:
    rows = load_corrected_rows(database, experiment_id)
    task_count = len({row["task_id"] for row in rows})
    repetition_count = len({row.get("repetition") for row in rows}) or 1
    paper_style = task_balanced_metric(rows, "combined_score", "process_status")
    process_dimensions = {
        dimension: task_balanced_metric(rows, dimension, "process_status")
        for dimension in (
            "tool_use_appropriate",
            "consistency",
            "robustness",
            "security_score",
            "process_score",
            "combined_score",
        )
    }
    task_scores = [
        {
            "task_id": task,
            "agent": agent,
            "language": language,
            "outcome_score": score,
        }
        for (task, agent, language), score in sorted(task_condition_scores(rows).items())
    ]
    status_counts: dict[str, int] = {}
    import sqlite3

    connection = sqlite3.connect(database)
    try:
        for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM run WHERE experiment_id = ? GROUP BY status",
            (experiment_id,),
        ):
            status_counts[str(status)] = int(count)
        expected = connection.execute(
            "SELECT COUNT(*) FROM run WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()[0]
        valid_grades = connection.execute(
            "SELECT COUNT(*) FROM grade g JOIN run r ON r.run_id=g.run_id WHERE r.experiment_id=? AND g.kind='task' AND g.status='completed'",
            (experiment_id,),
        ).fetchone()[0]
        valid_judges = connection.execute(
            "SELECT COUNT(*) FROM process_grade pg JOIN run r ON r.run_id=pg.run_id WHERE r.experiment_id=? AND pg.status='completed'",
            (experiment_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    report: dict[str, Any] = {
        "schema_version": 2,
        "provisional": True,
        "experiment_id": experiment_id,
        "study_design": f"{task_count} tasks x 3 languages x 3 harnesses x {repetition_count} repetition(s)",
        "cells": {
            "database_rows": expected,
            "completed_rows": len(rows),
            "oracle_gradable": valid_grades,
            "process_judged": valid_judges,
            "statuses": status_counts,
        },
        "task_balanced_outcome": task_balanced_outcome(rows),
        "paper_style_diagnostic": paper_style,
        "task_balanced_process_dimensions": process_dimensions,
        "task_scores": task_scores,
        "paired_oracle_deltas": paired_outcome_deltas(rows),
        "paired_oracle_bootstrap_95": bootstrap_paired_deltas(rows),
        "score_difference_distribution": score_difference_distribution(rows),
        "harness_interactions": harness_interactions(rows),
        "paired_binary_outcomes": paired_outcome_categories(rows),
        "error_conditioned_recovery": error_conditioned_recovery(database, rows),
        "usage_summary": usage_summary(rows),
        "trace_links": trace_links(database, rows),
        "notes": [
            "Hindi and Hinglish translations are unreviewed; findings are provisional.",
            "The same university model is used for agent calls and process judging; process scores are diagnostic, not independent validation.",
            "The 24 tasks are a fixed purposive subset of Harness-Bench and are not the paper's 106-task leaderboard.",
            "Repeated temperature-zero attempts are fresh executions but are not assumed to be independent random draws.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase I corrective study — provisional report",
        "",
        f"Experiment: `{experiment_id}`",
        "",
        "This report is provisional. Hindi and Hinglish translations remain unreviewed.",
        "",
        "## Cell completeness",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Database cells | {expected} |",
        f"| Completed cells | {len(rows)} |",
        f"| Oracle-gradable cells | {valid_grades} |",
        f"| Valid process judgments | {valid_judges} |",
        "",
        "## Task-balanced continuous oracle outcome",
        "",
        "| Harness/language | Mean outcome |",
        "|---|---:|",
    ]
    for key, value in sorted(report["task_balanced_outcome"].items()):
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(["", "## Task-balanced process/security diagnostics", "", "| Harness/language | Tool use | Consistency | Robustness | Security | Process | Combined |", "|---|---:|---:|---:|---:|---:|---:|"])
    condition_keys = sorted(report["task_balanced_outcome"])
    dimensions = report["task_balanced_process_dimensions"]
    for key in condition_keys:
        values = [dimensions.get(dimension, {}).get(key) for dimension in ("tool_use_appropriate", "consistency", "robustness", "security_score", "process_score", "combined_score")]
        rendered = ["—" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {key} | " + " | ".join(rendered) + " |")
    lines.extend(["", "## Paired Hindi/Hinglish deltas against English", "", "| Harness | Hindi − English | Hinglish − English |", "|---|---:|---:|"])
    for harness, values in sorted(report["paired_oracle_deltas"].items()):
        lines.append(f"| {harness} | {values.get('hindi', float('nan')):.4f} | {values.get('hinglish', float('nan')):.4f} |")
    lines.extend(["", "## 95% task-cluster bootstrap intervals for paired oracle deltas", "", "| Harness/language contrast | Lower | Upper |", "|---|---:|---:|"])
    for harness, values in sorted(report["paired_oracle_bootstrap_95"].items()):
        for language, interval in sorted(values.items()):
            lines.append(f"| {harness}: {language} − English | {interval['lower']:.4f} | {interval['upper']:.4f} |")
    lines.extend(["", "## Paired binary outcomes", ""])
    for harness, values in sorted(report["paired_binary_outcomes"].items()):
        lines.append(f"### {harness}\n")
        lines.extend(["| Outcome | Cells |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in sorted(values.items()))
        lines.append("")
    lines.extend(["## Harness interactions", "", "| Contrast | Value |", "|---|---:|"])
    for section, values in report["harness_interactions"].items():
        for key, value in sorted(values.items()):
            lines.append(f"| {section}: {key} | {value:.4f} |")
    lines.extend(["", "## Data products", "", f"- Task-level score rows: {len(report['task_scores'])}", f"- Paired score differences: {len(report['score_difference_distribution'])}", f"- Trace links: {len(report['trace_links'])}", f"- Machine-readable report: `{output.with_suffix('.json')}`"])
    lines.extend([
        "## Interpretation constraints",
        "",
        f"- This is the five-task, one-repetition pilot ({report['study_design']}); it is not the planned 648-cell main study.",
        "- Continuous oracle outcome is the primary language estimand; perfect completion is secondary.",
        "- The paper-style process/security aggregate is diagnostic because the agent model also judges it.",
        "- The same university model was used for agent calls and process judging; process scores are not independent validation.",
        "- No causal failure attribution is made without trace annotation.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
