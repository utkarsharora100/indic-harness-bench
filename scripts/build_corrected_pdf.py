from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _chart_paths(report: dict, directory: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    values = report.get("task_balanced_outcome", {})
    if values:
        path = directory / "outcome.png"
        labels = list(values)
        plt.figure(figsize=(10, 4.5))
        plt.bar(labels, [float(values[label]) for label in labels], color="#2C6EAA")
        plt.ylim(0, 1)
        plt.ylabel("Mean continuous oracle outcome")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(("Task-balanced continuous oracle outcome", path))

    task_rows = report.get("task_scores", [])
    if task_rows:
        tasks = sorted({row["task_id"] for row in task_rows})
        conditions = sorted({f"{row['agent']}:{row['language']}" for row in task_rows})
        lookup = {
            (row["task_id"], f"{row['agent']}:{row['language']}"): row["outcome_score"]
            for row in task_rows
        }
        matrix = [
            [lookup.get((task, condition), float("nan")) for condition in conditions]
            for task in tasks
        ]
        path = directory / "task-heatmap.png"
        height = max(5.0, 0.27 * len(tasks) + 1.8)
        plt.figure(figsize=(10, height))
        image = plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(image, label="Mean oracle outcome")
        plt.xticks(range(len(conditions)), conditions, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(tasks)), tasks, fontsize=7)
        plt.xlabel("Harness : language")
        plt.ylabel("Task")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(("Task-by-task oracle outcome heatmap", path))

    differences = report.get("score_difference_distribution", [])
    if differences:
        groups: dict[str, list[float]] = {}
        for row in differences:
            groups.setdefault(f"{row['agent']}:{row['language']}", []).append(
                float(row["difference"])
            )
        path = directory / "paired-differences.png"
        labels = sorted(groups)
        plt.figure(figsize=(10, 4.5))
        plt.boxplot([groups[label] for label in labels], labels=labels, showmeans=True)
        plt.axhline(0, color="#777777", linewidth=0.8)
        plt.ylabel("Task-level score difference from English")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(("Paired task-level score differences", path))
    return paths


def build(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase1-corrected-pdf-") as temporary:
        chart_paths = _chart_paths(report, Path(temporary))
        values = report.get("task_balanced_outcome", {})

        story = [
            Paragraph("Phase I corrective study", styles["Title"]),
            Paragraph("Language × harness evaluation on a fixed 24-task Harness-Bench subset", styles["Heading2"]),
            Paragraph(
                "Provisional report. Hindi and Hinglish translations remain unreviewed. "
                "The earlier 216-cell experiment is superseded and is not pooled with this study.",
                styles["BodyText"],
            ),
            Spacer(1, 0.16 * inch),
            Paragraph(f"Experiment: {report.get('experiment_id', 'unknown')}", styles["BodyText"]),
            Paragraph(
                "The corrected protocol follows the pinned Harness-Bench task/oracle contract and records "
                "continuous completion, trace-based process dimensions, and security separately. The same "
                "university model is used as the process judge, so that score is diagnostic rather than independent.",
                styles["BodyText"],
            ),
            Spacer(1, 0.12 * inch),
        ]
        for title, chart_path in chart_paths:
            story.extend([
                Paragraph(title, styles["Heading2"]),
                Image(str(chart_path), width=7.0 * inch, height=3.15 * inch),
                Spacer(1, 0.12 * inch),
            ])
        table = [["Harness/language", "Mean outcome"]]
        table.extend([[label, f"{float(value):.4f}"] for label, value in sorted(values.items())])
        result_table = Table(table, colWidths=[4.8 * inch, 1.5 * inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E0E7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ]))
        story.extend([Paragraph("Continuous oracle outcome", styles["Heading2"]), result_table])
        story.extend([
            Spacer(1, 0.16 * inch),
            Paragraph("Methodology and interpretation", styles["Heading2"]),
            Paragraph(
                "Primary estimands are task-cluster paired Hindi−English and Hinglish−English continuous "
                "oracle differences within each harness. Three temperature-zero attempts are averaged within "
                "task × language × harness; tasks, not attempts, are the bootstrap clusters. The paper-style "
                "aggregate is completion × mean(process dimensions) × security. The original Harness-Bench "
                "paper used 106 tasks, six harnesses, eight model backends, and an external judge; these results "
                "are therefore not numerically comparable to its leaderboard.",
                styles["BodyText"],
            ),
            Spacer(1, 0.12 * inch),
            Paragraph(
                f"The machine-readable report contains {len(report.get('trace_links', []))} per-cell trace/result links. "
                "Missing usage, grades, or process judgments remain missing; no missing value is imputed as zero or one.",
                styles["BodyText"],
            ),
            Spacer(1, 0.12 * inch),
            Paragraph("Paper reference: arXiv:2605.27922, Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows.", styles["BodyText"]),
            PageBreak(),
            Paragraph("Protocol deviations from the original Harness-Bench paper", styles["Heading2"]),
        ])
        def table_cell(value: str) -> Paragraph:
            return Paragraph(value, styles["BodyText"])

        deviation_table = Table([
            [table_cell("Dimension"), table_cell("Corrected Phase I"), table_cell("Original paper")],
            [table_cell("Task set"), table_cell("Fixed purposive 24-task subset"), table_cell("106 tasks")],
            [table_cell("Harnesses"), table_cell("ReAct, native NanoBot, native OpenClaw"), table_cell("Six configurable harnesses")],
            [table_cell("Models"), table_cell("One university GPU model"), table_cell("Eight model backends")],
            [table_cell("Process judge"), table_cell("Same model; diagnostic, non-independent"), table_cell("External judge")],
            [table_cell("Primary language estimand"), table_cell("Paired task-balanced continuous oracle deltas"), table_cell("Not a language comparison")],
        ], colWidths=[1.35 * inch, 2.65 * inch, 2.65 * inch])
        deviation_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E0E7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ]))
        story.append(deviation_table)
        doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch)
        doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the corrected Phase I PDF")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
