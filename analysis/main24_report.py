"""Read-only report generator for the frozen Phase I main24 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.corrected import bootstrap_paired_deltas, task_balanced_metric, task_condition_scores
from runner.config import ExperimentConfig
from runner.hybrid_outcome import _independent_tests, _packet
from runner.inference import load_env_file
from runner.outcome_judge import read_workspace_archive
from runner.outcome_v2 import _scan_for_secrets, verify_frozen_inputs

LANGUAGES = ("english", "hindi", "hinglish")
HARNESSES = ("react", "nanobot", "openclaw")


class Main24ReportError(RuntimeError):
    pass


def _ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise Main24ReportError(f"Missing input store: {path.name}")
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _read_rows(database: Path, outcomes: Path, experiment_id: str) -> list[dict[str, Any]]:
    source = _ro(database)
    judged = _ro(outcomes)
    try:
        runs = source.execute(
            """SELECT r.*,g.score AS oracle_score,g.status AS grade_status,
                      pg.status AS process_status,pg.tool_use_appropriate,pg.consistency,
                      pg.robustness,pg.process_score,pg.security_score,pg.combined_score
               FROM run r JOIN grade g ON g.run_id=r.run_id AND g.test_name='task_grader'
                 AND g.kind='task_grader'
               LEFT JOIN process_grade pg ON pg.run_id=r.run_id
               WHERE r.experiment_id=? ORDER BY r.task_id,r.agent,r.language""",
            (experiment_id,),
        ).fetchall()
        ratings = {row["cell_id"]: dict(row) for row in judged.execute("SELECT * FROM judgment")}
    finally:
        source.close()
        judged.close()
    if len(runs) != 216:
        raise Main24ReportError(f"Expected 216 graded agent cells, found {len(runs)}")
    rows = []
    for run in runs:
        row = dict(run)
        rating = ratings.get(row["cell_id"])
        if not rating or rating["status"] != "completed" or rating["outcome_score"] is None:
            raise Main24ReportError("A primary LLM outcome judgment is missing")
        row.update(
            {
                "outcome_score": rating["outcome_score"],
                "semantic_level": rating["semantic_level"],
                "judgment_status": rating["status"],
                "oracle_disagreement": rating["outcome_score"] - row["oracle_score"],
                "outcome_judge_details": rating["detail_json"],
                "outcome_packet_sha256": rating["packet_sha256"],
                "outcome_archive_sha256": rating["archive_sha256"],
                "outcome_workspace_sha256": rating["workspace_sha256"],
                "outcome_trace_sha256": rating["trace_sha256"],
            }
        )
        row["hybrid_diagnostic"] = (
            row["outcome_score"] * row["process_score"] * row["security_score"]
            if row["process_score"] is not None and row["security_score"] is not None
            else None
        )
        rows.append(row)
    if len(ratings) != 216:
        raise Main24ReportError("Outcome store includes missing or unexpected main-study cells")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    condition = []
    by_task = task_condition_scores(rows, "outcome_score", "judgment_status")
    for harness in HARNESSES:
        for language in LANGUAGES:
            task_values = [
                score
                for (task, agent, lang), score in by_task.items()
                if agent == harness and lang == language
            ]
            selected = [
                row for row in rows if row["agent"] == harness and row["language"] == language
            ]
            fields = (
                "oracle_score",
                "outcome_score",
                "semantic_level",
                "initial_prompt_tokens",
                "total_tokens",
                "agent_time",
                "end_to_end_time",
                "tool_calls",
                "failed_tool_calls",
                "process_score",
                "security_score",
                "hybrid_diagnostic",
            )
            record: dict[str, Any] = {
                "harness": harness,
                "language": language,
                "task_count": len(task_values),
            }
            for field in fields:
                values = [float(row[field]) for row in selected if row.get(field) is not None]
                record[f"mean_{field}"] = mean(values) if values else None
                record[f"n_{field}"] = len(values)
            condition.append(record)
    paired_rows = []
    paired_source = [{**row, "process_status": row["judgment_status"]} for row in rows]
    deltas = bootstrap_paired_deltas(
        paired_source, metric="outcome_score", repetitions=20_000, seed=1701
    )
    means = task_balanced_metric(paired_source, "outcome_score", "grade_status")
    for harness in HARNESSES:
        for language in ("hindi", "hinglish"):
            value = deltas.get(harness, {}).get(language)
            delta_values = [
                values[language] - values["english"]
                for (task, agent, lang), score in by_task.items()
                if agent == harness and lang == language
                for values in [
                    {
                        "english": by_task.get((task, harness, "english")),
                        language: score,
                    }
                ]
                if values["english"] is not None
            ]
            paired_rows.append(
                {
                    "harness": harness,
                    "contrast": f"{language}-English",
                    "mean_paired_difference": mean(delta_values) if delta_values else None,
                    "n_tasks": value["n_tasks"] if value else 0,
                    "bootstrap_95_lower": value["lower"] if value else None,
                    "bootstrap_95_upper": value["upper"] if value else None,
                }
            )
    categories = defaultdict(list)
    for row in rows:
        categories[row.get("task_category", row["task_id"])].append(row["outcome_score"])
    extra = {
        "n_agent_cells": len(rows),
        "n_primary_scores": sum(row["outcome_score"] is not None for row in rows),
        "n_missing_usage": sum(row["total_tokens"] is None for row in rows),
        "n_missing_process": sum(row["process_score"] is None for row in rows),
        "n_missing_security": sum(row["security_score"] is None for row in rows),
        "task_balanced_condition_means": means,
    }
    return condition, paired_rows, extra


def _plots(rows: list[dict[str, Any]], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    means = defaultdict(list)
    for row in rows:
        means[(row["agent"], row["language"])].append(float(row["outcome_score"]))
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    x = np.arange(len(HARNESSES))
    width = 0.24
    colors = ("#315b7d", "#36a18b", "#e38b45")
    for idx, language in enumerate(LANGUAGES):
        values = [mean(means[(harness, language)]) for harness in HARNESSES]
        ax.bar(x + (idx - 1) * width, values, width, label=language.title(), color=colors[idx])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Task-balanced LLM outcome score")
    ax.set_xticks(x, [name.title() for name in HARNESSES])
    ax.set_title("Performance by harness and prompt language")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = out / "opening_harness_language_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    task_scores = task_condition_scores(rows, "outcome_score", "judgment_status")
    ordered_tasks = sorted({key[0] for key in task_scores})
    labels = [(harness, language) for harness in HARNESSES for language in LANGUAGES]
    matrix = np.array(
        [
            [task_scores.get((task, harness, lang), np.nan) for harness, lang in labels]
            for task in ordered_tasks
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 7.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(len(ordered_tasks)), ordered_tasks, fontsize=7)
    ax.set_xticks(
        range(len(labels)),
        [f"{harness}\n{language}" for harness, language in labels],
        fontsize=7,
    )
    ax.set_title("Task-by-condition LLM outcome heatmap")
    fig.colorbar(image, ax=ax, label="Score")
    fig.tight_layout()
    path = out / "task_condition_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, harness in zip(axes, HARNESSES, strict=True):
        for task in ordered_tasks:
            values = [task_scores.get((task, harness, lang)) for lang in LANGUAGES]
            if all(value is not None for value in values):
                ax.plot(range(3), values, color="#78909c", alpha=0.45, linewidth=0.8)
        ax.set_title(harness.title())
        ax.set_xticks(range(3), [lang.title() for lang in LANGUAGES], rotation=20)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Hybrid outcome score; one line per task")
    fig.suptitle("Paired task-level language profiles")
    fig.tight_layout()
    path = out / "paired_language_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for harness in HARNESSES:
        subset = [row for row in rows if row["agent"] == harness]
        ax.scatter(
            [row["oracle_score"] for row in subset],
            [row["outcome_score"] for row in subset],
            label=harness.title(),
            alpha=0.65,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555", linewidth=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Deterministic oracle",
        ylabel="Hybrid primary outcome",
        title="Oracle comparison (audit view)",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out / "hybrid_vs_oracle.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _review_packet(config: ExperimentConfig, rows: list[dict[str, Any]], out: Path) -> None:
    rng = random.Random(1701)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    condition_counts: Counter[tuple[str, str]] = Counter()
    selected = []
    for task in tasks:
        candidates = by_task[task][:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: condition_counts[(row["agent"], row["language"])])
        chosen = candidates[0]
        condition_counts[(chosen["agent"], chosen["language"])] += 1
        selected.append(chosen)
    packet_dir = out / "human_review"
    packet_dir.mkdir(parents=True, exist_ok=True)
    with (packet_dir / "blinded_packet.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, 1):
            archive_root = (config.root / config.storage["results_dir"] / "workspaces").resolve()
            archive_path = Path(json.loads(row["metadata_json"])["workspace_archive"])
            if archive_path.is_symlink() or not archive_path.is_file():
                raise Main24ReportError("A selected review archive is missing or unsafe")
            archive_path = archive_path.resolve()
            if not archive_path.is_relative_to(archive_root):
                raise Main24ReportError(
                    "A review archive escapes the main-study workspace directory"
                )
            files, _, _ = read_workspace_archive(archive_path)
            source = config.task_root / row["task_id"] / "source"
            validation = _independent_tests(row["task_id"], files, str(config.sandbox["image"]))
            packet = _packet(row["task_id"], source, files, validation)
            review = {
                "review_id": f"R{index:02d}",
                "task_id": row["task_id"],
                "canonical_english_question": packet["canonical_english_question"],
                "reference_answer": packet["reference_answer_not_agent_work"],
                "submitted_workspace": packet["submitted_workspace"],
                "rubric_scale_0_to_4": (
                    "0 absent/contradicted; 1 major errors; 2 useful partial; "
                    "3 minor omissions; 4 fully satisfies"
                ),
            }
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")
    (packet_dir / "blank_review_form.csv").write_text(
        "review_id,criterion,level_0_to_4,evidence_path,reviewer_note\n",
        encoding="utf-8",
    )
    (packet_dir / "unblinding_key.json").write_text(
        json.dumps(
            {
                f"R{index:02d}": {
                    "cell_id": row["cell_id"],
                    "language": row["language"],
                    "harness": row["agent"],
                }
                for index, row in enumerate(selected, 1)
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_report(config_path: Path) -> dict[str, Any]:
    config = ExperimentConfig.load(config_path)
    root = config.root
    database = root / config.storage["database"]
    outcomes = root / "data/phase1/corrected/main24-v1/outcomes-v3.sqlite"
    output_dir = root / "data/phase1/corrected/main24-v1/report"
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs(root, root / "data/phase1/corrected/pilot-v14/runs.sqlite")
    rows = _read_rows(database, outcomes, config.experiment_id)
    manifest = json.loads(
        (root / config.experiment["dataset_manifest"]).read_text(encoding="utf-8")
    )
    categories = {
        item["task_id"]: item.get("category", "uncategorized") for item in manifest["tasks"]
    }
    for row in rows:
        row["task_category"] = categories.get(row["task_id"], "uncategorized")
    condition, paired, counts = _summary(rows)
    _write_csv(output_dir / "cells.csv", rows)
    _write_csv(output_dir / "condition_summary.csv", condition)
    _write_csv(output_dir / "paired_language_contrasts.csv", paired)
    category_rows = []
    for category in sorted(set(categories.values())):
        selected = [row["outcome_score"] for row in rows if row["task_category"] == category]
        category_rows.append(
            {
                "category": category,
                "task_count": len({r["task_id"] for r in rows if r["task_category"] == category}),
                "cell_count": len(selected),
                "mean_outcome_score": mean(selected) if selected else None,
            }
        )
    _write_csv(output_dir / "category_summary.csv", category_rows)
    plot_paths = _plots(rows, output_dir / "figures")
    _review_packet(config, rows, output_dir)
    trace_index = []
    trace_root = (database.parent / "traces").resolve()
    archive_root = (database.parent / "results" / "workspaces").resolve()
    for row in rows:
        trace_path = Path(row["trace_path"] or "")
        if trace_path.is_symlink() or not trace_path.is_file():
            raise Main24ReportError("A run trace is missing or unsafe")
        trace_path = trace_path.resolve()
        if not trace_path.is_relative_to(trace_root):
            raise Main24ReportError("A run trace escapes the main-study trace directory")
        archive_path = Path(json.loads(row["metadata_json"])["workspace_archive"])
        if archive_path.is_symlink() or not archive_path.is_file():
            raise Main24ReportError("A run workspace archive is missing or unsafe")
        archive_path = archive_path.resolve()
        if not archive_path.is_relative_to(archive_root):
            raise Main24ReportError("A workspace archive escapes the main-study result directory")
        trace_index.append(
            {
                "cell_id": row["cell_id"],
                "task_id": row["task_id"],
                "language": row["language"],
                "harness": row["agent"],
                "trace": str(trace_path),
                "workspace_archive": str(archive_path),
            }
        )
    _write_csv(output_dir / "trace_index.csv", trace_index)
    summary = {
        "experiment_id": config.experiment_id,
        "matrix": "24 tasks x 3 languages x 3 harnesses x 1 execution = 216 cells",
        "task_balanced_condition_summary": condition,
        "paired_language_contrasts": paired,
        "category_summary": category_rows,
        "coverage": counts,
        "interpretation_limitations": [
            "one execution per condition; purposively selected task subset",
            "Hindi/Hinglish translations remain unreviewed",
            "same university model provides semantic outcome judgments",
            "findings are exploratory, not proof of equivalence or causal superiority",
        ],
        "preserved_pilot_hashes": frozen,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pdf_path = root / "output/pdf/indic_harness_phase1_main24_v1.pdf"
    _build_pdf(summary, plot_paths, pdf_path, trace_index)
    secret_file = root / config.inference.get("env_file", ".env.uni-gpu.local")
    private = load_env_file(secret_file)
    model_manifest = json.loads((root / config.inference["manifest"]).read_text(encoding="utf-8"))
    secrets = tuple(
        value
        for value in (
            private.get("INDIC_UNI_GPU_BASE_URL"),
            private.get("INDIC_UNI_GPU_API_KEY"),
            model_manifest.get("resolved_model"),
        )
        if value
    )
    generated_artifacts = [path for path in output_dir.rglob("*") if path.is_file()]
    _scan_for_secrets([pdf_path, *generated_artifacts], secrets)
    verify_frozen_inputs(root, root / "data/phase1/corrected/pilot-v14/runs.sqlite")
    return {"pdf": str(pdf_path), "tables": str(output_dir), "cells": len(rows)}


def _build_pdf(
    summary: dict[str, Any],
    plots: list[Path],
    path: Path,
    trace_index: list[dict[str, Any]],
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Phase I main study: 24-task multilingual harness comparison", styles["Title"]),
        Paragraph("Provisional research report · 216 single-execution cells", styles["Heading2"]),
        Paragraph(
            "Exploratory only: purposively selected tasks, unreviewed translations, "
            "and the same model used for semantic judging.",
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * inch),
        Paragraph("Executive view: harness × language", styles["Heading1"]),
        Image(str(plots[0]), width=7.1 * inch, height=3.4 * inch),
    ]
    rows = [["Harness", "Language", "Tasks", "LLM mean", "Oracle audit mean", "LLM level"]]
    for item in summary["task_balanced_condition_summary"]:
        rows.append(
            [
                item["harness"],
                item["language"],
                item["task_count"],
                f"{item['mean_outcome_score']:.3f}",
                f"{item['mean_oracle_score']:.3f}",
                f"{item['mean_semantic_level']:.2f}",
            ]
        )
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21445e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f5")]),
            ]
        )
    )
    story.extend(
        [
            table,
            PageBreak(),
            Paragraph("Task-level paired outcomes", styles["Heading1"]),
            Image(str(plots[1]), width=7.1 * inch, height=4.7 * inch),
            Paragraph(
                "Each heatmap row is a selected task; columns are harness-language conditions.",
                styles["BodyText"],
            ),
            Image(str(plots[2]), width=7.1 * inch, height=2.8 * inch),
            PageBreak(),
            Paragraph("Paired language contrasts", styles["Heading1"]),
        ]
    )
    paired = [["Harness", "Contrast", "Mean paired Δ", "Tasks", "95% task-cluster interval"]]
    for item in summary["paired_language_contrasts"]:
        interval = (
            "missing"
            if item["bootstrap_95_lower"] is None
            else f"[{item['bootstrap_95_lower']:.3f}, {item['bootstrap_95_upper']:.3f}]"
        )
        delta = (
            "missing"
            if item["mean_paired_difference"] is None
            else f"{item['mean_paired_difference']:.3f}"
        )
        paired.append([item["harness"], item["contrast"], delta, item["n_tasks"], interval])
    ptable = Table(paired, repeatRows=1)
    ptable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21445e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            ptable,
            Spacer(1, 0.2 * inch),
            Image(str(plots[3]), width=5.9 * inch, height=4.5 * inch),
            PageBreak(),
            Paragraph("Coverage, categories, and interpretation", styles["Heading1"]),
        ]
    )
    cats = [["Category", "Tasks", "Cells", "Mean LLM score"]]
    for item in summary["category_summary"]:
        cats.append(
            [
                item["category"],
                item["task_count"],
                item["cell_count"],
                "missing"
                if item["mean_outcome_score"] is None
                else f"{item['mean_outcome_score']:.3f}",
            ]
        )
    ctable = Table(cats, repeatRows=1)
    ctable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21445e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(ctable)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(json.dumps(summary["coverage"], indent=2), styles["Code"]))
    for note in summary["interpretation_limitations"]:
        story.append(Paragraph(f"• {note}", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Per-cell traces, archived workspaces, criteria, and the blinded review "
            "packet are indexed in the adjacent machine-readable tables.",
            styles["BodyText"],
        )
    )
    story.extend(
        [
            PageBreak(),
            Paragraph("Trace and workspace links", styles["Heading1"]),
            Paragraph(
                "Open a JSONL trace or archived final workspace from this local report. "
                "Condition labels appear only for traceability.",
                styles["BodyText"],
            ),
        ]
    )
    links = [["Cell", "Task", "Condition", "Trace", "Workspace"]]
    for item in trace_index:
        trace_uri = Path(item["trace"]).as_uri()
        workspace_uri = Path(item["workspace_archive"]).as_uri()
        links.append(
            [
                item["cell_id"],
                item["task_id"],
                f"{item['harness']} / {item['language']}",
                Paragraph(f'<link href="{trace_uri}">trace</link>', styles["BodyText"]),
                Paragraph(f'<link href="{workspace_uri}">workspace</link>', styles["BodyText"]),
            ]
        )
    link_table = LongTable(
        links,
        colWidths=[1.25 * inch, 1.8 * inch, 1.25 * inch, 0.7 * inch, 0.8 * inch],
        repeatRows=1,
    )
    link_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21445e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.2, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 5.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(link_table)
    SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    ).build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase1.corrected.main24-v1.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.config), indent=2))


if __name__ == "__main__":
    main()
