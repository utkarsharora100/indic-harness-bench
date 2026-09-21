from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


LANGUAGES = ("english", "hindi", "hinglish")
LANGUAGE_LABELS = {"english": "English", "hindi": "Hindi", "hinglish": "Hinglish"}
CATEGORY_LABELS = {
    "software_engineering": "Software engineering",
    "data_file_processing": "Data and file analysis",
    "shell_tool_use": "Workspace and shell",
    "office_business": "Office workflows",
    "knowledge_retrieval": "Evidence and retrieval",
    "multi_step_workflow": "Multistep workflows",
}
PALETTE = {
    "navy": "#17324D",
    "blue": "#2C6EAA",
    "teal": "#1F8A8A",
    "orange": "#D97925",
    "ink": "#202B36",
    "muted": "#5D6B78",
    "line": "#D8E0E7",
    "pale": "#F3F6F8",
    "green": "#3A8F5B",
    "red": "#B64B4B",
}


def _hex(value: str) -> colors.Color:
    return colors.HexColor(value)


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "missing"
    return f"{float(value):,.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "missing"
    return f"{float(value) * 100:.{digits}f}%"


def _safe(text: Any) -> str:
    return escape(str(text))


def _read_data(root: Path, database: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = yaml.safe_load((root / "benchmark/task_selection.yaml").read_text(encoding="utf-8"))
    selected = selection.get("tasks", [])
    task_info = {
        str(item["task_id"]): {
            "category": str(item.get("category", "")),
            "source_sha256": str(item.get("source_sha256", "")),
        }
        for item in selected
    }

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(
            """
            SELECT r.*, g.passed AS grade_passed, g.details AS grade_details
            FROM run r
            LEFT JOIN grade g ON g.run_id = r.run_id AND g.test_name = 'task_grader'
            WHERE r.experiment_id = ?
            ORDER BY r.task_id, r.language, r.repetition
            """,
            (report["experiment_id"],),
        )]
        attempt_counts = {
            status: count
            for status, count in connection.execute(
                """
                SELECT a.status, COUNT(*)
                FROM attempt a JOIN run r ON r.run_id = a.run_id
                WHERE r.experiment_id = ?
                GROUP BY a.status
                """,
                (report["experiment_id"],),
            )
        }
        event_count = connection.execute(
            """
            SELECT COUNT(*) FROM event e JOIN run r ON r.run_id = e.run_id
            WHERE r.experiment_id = ?
            """,
            (report["experiment_id"],),
        ).fetchone()[0]
        failed_tool_events = connection.execute(
            """
            SELECT COUNT(*) FROM event e JOIN run r ON r.run_id = e.run_id
            WHERE r.experiment_id = ? AND e.event_type = 'tool_call'
              AND json_extract(e.result_json, '$.ok') = 0
            """,
            (report["experiment_id"],),
        ).fetchone()[0]
    finally:
        connection.close()

    for row in rows:
        row["category"] = task_info.get(row["task_id"], {}).get("category", "")
        row["oracle_score"] = None
        details = row.get("grade_details") or ""
        try:
            details_payload = json.loads(details)
            stdout = details_payload.get("stdout", "")
            oracle_payload = json.loads(stdout)
            if isinstance(oracle_payload, dict) and "outcome_score" in oracle_payload:
                row["oracle_score"] = float(oracle_payload["outcome_score"])
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_groups[row["task_id"]].append(row)
        category_groups[row["category"]].append(row)

    task_summary = []
    for task_id in [str(item["task_id"]) for item in selected]:
        group = task_groups[task_id]
        summary = {
            "task_id": task_id,
            "category": task_info[task_id]["category"],
            "n": len(group),
        }
        for language in LANGUAGES:
            values = [float(row["success"]) for row in group if row["language"] == language]
            scores = [row["oracle_score"] for row in group if row["language"] == language and row["oracle_score"] is not None]
            summary[f"{language}_rate"] = mean(values) if values else None
            summary[f"{language}_score"] = mean(scores) if scores else None
        task_summary.append(summary)

    category_summary = []
    for category in CATEGORY_LABELS:
        group = category_groups[category]
        task_count = len({row["task_id"] for row in group})
        summary = {"category": category, "tasks": task_count}
        for language in LANGUAGES:
            values = [float(row["success"]) for row in group if row["language"] == language]
            summary[f"{language}_rate"] = mean(values) if values else None
        category_summary.append(summary)

    return {
        "report": report,
        "selection": selection,
        "rows": rows,
        "task_summary": task_summary,
        "category_summary": category_summary,
        "attempt_counts": attempt_counts,
        "event_count": event_count,
        "failed_tool_events": failed_tool_events,
    }


def _chart_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_success_chart(path: Path, report: dict[str, Any]) -> None:
    _chart_style()
    values = [float(report["task_balanced_success"].get(language, 0.0)) * 100 for language in LANGUAGES]
    intervals = report["bootstrap_95_task_cluster"]
    lower = [values[i] - float(intervals[language]["lower"]) * 100 for i, language in enumerate(LANGUAGES)]
    upper = [float(intervals[language]["upper"]) * 100 - values[i] for i, language in enumerate(LANGUAGES)]
    labels = [LANGUAGE_LABELS[language] for language in LANGUAGES]
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=180)
    bars = ax.bar(labels, values, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]], width=0.55)
    ax.errorbar(range(3), values, yerr=[lower, upper], fmt="none", ecolor=PALETTE["ink"], elinewidth=1.2, capsize=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Task-balanced success (%)")
    ax.set_title("Strict success is identical across instruction languages", loc="left", pad=12, fontweight="bold")
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["line"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.text(0.0, -0.23, "Whiskers: 95% task-cluster bootstrap interval", transform=ax.transAxes, color=PALETTE["muted"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_metrics_chart(path: Path, report: dict[str, Any]) -> None:
    _chart_style()
    metrics = report["secondary_metrics"]
    panels = [
        ("Total tokens", "total_tokens", 1000, "Thousands"),
        ("End-to-end time", "end_to_end_time_seconds", 1, "Seconds"),
        ("Tool calls", "tool_calls", 1, "Calls"),
        ("Failed tool calls", "failed_tool_calls", 1, "Calls"),
    ]
    colors_for_lang = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), dpi=180)
    x = list(range(3))
    for ax, (title, key, scale, ylabel) in zip(axes.flat, panels):
        values = [float(metrics[key].get(language) or 0.0) / scale for language in LANGUAGES]
        bars = ax.bar([0, 1, 2], values, color=colors_for_lang, width=0.58)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, ["EN", "HI", "Hinglish"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=PALETTE["line"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(PALETTE["line"])
        top = max(values) if values else 1
        ax.set_ylim(0, top * 1.24 if top > 0 else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.04, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Operational cost and interaction profile", x=0.08, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_task_heatmap(path: Path, task_summary: list[dict[str, Any]]) -> None:
    _chart_style()
    labels = [item["task_id"] for item in task_summary]
    values = [[float(item[f"{language}_rate"] or 0.0) * 100 for language in LANGUAGES] for item in task_summary]
    fig, ax = plt.subplots(figsize=(7.0, 8.8), dpi=180)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks([0, 1, 2], ["English", "Hindi", "Hinglish"])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Instruction language")
    ax.set_title("Task-level strict success pattern", loc="left", pad=12, fontweight="bold")
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            label_color = "white" if value >= 55 else PALETTE["ink"]
            ax.text(column_index, row_index, f"{value:.0f}%", ha="center", va="center", color=label_color, fontsize=8)
    for index in range(1, len(task_summary)):
        if task_summary[index]["category"] != task_summary[index - 1]["category"]:
            ax.axhline(index - 0.5, color="white", linewidth=2)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03, label="Success rate (%)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=_hex(PALETTE["navy"]), alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=_hex(PALETTE["muted"]), spaceAfter=15),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=_hex(PALETTE["navy"]), spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=_hex(PALETTE["navy"]), spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=_hex(PALETTE["ink"]), spaceAfter=6),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10, textColor=_hex(PALETTE["muted"]), spaceAfter=4),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.8, leading=10, textColor=_hex(PALETTE["muted"]), alignment=TA_LEFT, spaceBefore=3, spaceAfter=8),
        "kpi": ParagraphStyle("KPI", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=_hex(PALETTE["navy"]), alignment=TA_CENTER),
        "kpi_label": ParagraphStyle("KPILabel", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=_hex(PALETTE["muted"]), alignment=TA_CENTER),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=9.4, textColor=_hex(PALETTE["ink"])),
        "table_header": ParagraphStyle("TableHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.6, textColor=colors.white, alignment=TA_LEFT),
        "table_header_center": ParagraphStyle("TableHeaderCenter", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.6, textColor=colors.white, alignment=TA_CENTER),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_safe(text), style)


def _styled_table(data: list[list[Any]], widths: list[float], header_rows: int = 1, repeat_rows: int = 1, font_size: float = 7.6) -> LongTable:
    table = LongTable(data, colWidths=widths, repeatRows=repeat_rows, splitByRow=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), _hex(PALETTE["navy"])),
                ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, header_rows), (-1, -1), 0.35, _hex(PALETTE["line"])),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, _hex(PALETTE["pale"])]),
            ]
        )
    )
    return table


def _figure_caption(number: str, title: str, description: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [_p(f"Figure {number}. {title}. {description}", styles["caption"])]


def _build_story(root: Path, data: dict[str, Any], figure_paths: dict[str, Path]) -> list[Any]:
    styles = _styles()
    report = data["report"]
    rows = data["rows"]
    story: list[Any] = []
    story.append(_p("Phase I: Controlled English, Hindi, and Hinglish Benchmark", styles["title"]))
    if report.get("experiment_id") == "phase1_language_comparison":
        story.append(_p("SUPERSEDED FOR INFERENCE — use the corrected language × harness study instead. This historical report is retained for audit only.", styles["subtitle"]))
    story.append(_p("Research findings report | Provisional pending translation review | Generated from the completed 216-cell study", styles["subtitle"]))

    kpi_data = [
        [_p("216", styles["kpi"]), _p("27", styles["kpi"]), _p("3", styles["kpi"]), _p("0", styles["kpi"])],
        [_p("Gradable cells", styles["kpi_label"]), _p("Strict successes", styles["kpi_label"]), _p("Instruction languages", styles["kpi_label"]), _p("Infrastructure gaps", styles["kpi_label"])],
    ]
    kpi_table = Table(kpi_data, colWidths=[1.7 * inch] * 4, rowHeights=[0.42 * inch, 0.34 * inch])
    kpi_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(PALETTE["pale"])), ("BOX", (0, 0), (-1, -1), 0.5, _hex(PALETTE["line"])), ("INNERGRID", (0, 0), (-1, -1), 0.35, _hex(PALETTE["line"])), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(_p("Executive finding", styles["h1"]))
    story.append(_p("The university-GPU ReAct agent achieved the same strict task success rate in all three instruction languages: 9 of 72 cells per language, or 12.5%. The observed language deltas are therefore zero in this study. The result is best interpreted as evidence of no measured language effect under this fixed model, agent, task sample, and translation version - not as evidence that the languages are universally equivalent.", styles["body"]))
    story.append(Image(str(figure_paths["success"]), width=6.95 * inch, height=3.45 * inch))
    story.extend(_figure_caption("1", "Strict success by language", "Bars show task-balanced success; whiskers show 95% task-cluster bootstrap intervals.", styles))

    story.append(_p("How to read this report", styles["h2"]))
    story.append(_p("A cell is one task-language-repetition condition. Success uses each upstream task's strict expected outcome threshold or judge exit code. Infrastructure failures are excluded from success rates; the final database contains none. Hindi and Latin-script Hinglish overlays were authored before execution but are still marked unreviewed, so all findings are provisional.", styles["body"]))

    story.append(PageBreak())
    story.append(_p("1. Study design and provenance", styles["h1"]))
    design_rows = [
        [_p("Item", styles["table_header"]), _p("Fixed value", styles["table_header"])],
        [_p("Benchmark source", styles["table"]), _p("Harness-Bench pinned at commit 1025086a446653702b80cfb48babbeec35db6b2c; 24 local task definitions and source hashes.", styles["table"])],
        [_p("Matrix", styles["table"]), _p("24 tasks x 3 languages x 3 repetitions = 216 stable cells; shuffle seed 17; sequential execution.", styles["table"])],
        [_p("Languages", styles["table"]), _p("English, Hindi, and natural Latin-script Hinglish. English prompts were preserved verbatim.", styles["table"])],
        [_p("Agent and generation", styles["table"]), _p("One ReAct agent; temperature 0; top-p 1; maximum 2,048 generated tokens per model call; maximum 40 agent steps; task-specific timeouts.", styles["table"])],
        [_p("Inference", styles["table"]), _p("University GPU OpenAI-compatible endpoint. The served model identity and tool-call check were pinned locally; the private endpoint, key, and server-backed model identifier are intentionally omitted from this report.", styles["table"])],
        [_p("Isolation", styles["table"]), _p("Fresh fixtures per cell; agent shell commands in a network-disabled Docker container; final workspace graded in a separate container with the oracle unavailable to the agent.", styles["table"])],
        [_p("Image", styles["table"]), _p("indic-harness-phase1:2026-09-21; digest 21ecab43c35b707512e8ad137343fb1fbea45d1059c159d26b6947affe8f74af.", styles["table"])],
    ]
    story.append(_styled_table(design_rows, [1.35 * inch, 5.55 * inch]))
    story.append(Spacer(1, 0.14 * inch))
    story.append(_p("Preflight and pilot", styles["h2"]))
    story.append(_p("The five-task pilot covered 15 runs across the same three languages and passed infrastructure checks. It identified and corrected two runner issues before the main study: provider-side malformed tool-call JSON was recorded as a model/tool error, and large or exception-bearing oracle output was preserved as a gradable score rather than lost as infrastructure. The main study was then resumed from stable cell identities without duplicating completed cells.", styles["body"]))

    story.append(_p("Completion audit", styles["h2"]))
    audit_rows = [
        [_p("Audit item", styles["table_header"]), _p("Observed", styles["table_header"]), _p("Interpretation", styles["table_header"])],
        [_p("Stable database cells", styles["table"]), _p("216 completed", styles["table"]), _p("Meets the Phase I completion criterion.", styles["table"])],
        [_p("Grades", styles["table"]), _p("216", styles["table"]), _p("Every completed cell has a grade record.", styles["table"])],
        [_p("Trace and result files", styles["table"]), _p("216 JSONL traces + 216 result JSON files", styles["table"]), _p("One per stable cell; indexed locally by cell_id.", styles["table"])],
        [_p("Event records", styles["table"]), _p(f"{data['event_count']:,}", styles["table"]), _p("Tool arguments/results and model-call metadata retained without private chain-of-thought.", styles["table"])],
        [_p("Failed tool results", styles["table"]), _p(f"{data['failed_tool_events']:,} across retained attempts", styles["table"]), _p("Includes command exit failures; final-cell counters exclude superseded retry attempts.", styles["table"])],
        [_p("Secret scan", styles["table"]), _p("No private marker in tracked files or experiment traces", styles["table"]), _p("Private runtime values remain in ignored local storage.", styles["table"])],
    ]
    story.append(_styled_table(audit_rows, [1.45 * inch, 1.85 * inch, 3.6 * inch]))

    story.append(PageBreak())
    story.append(_p("2. Main outcomes", styles["h1"]))
    outcomes = report["task_balanced_success"]
    intervals = report["bootstrap_95_task_cluster"]
    counts = report["cell_success_counts"]
    main_rows = [[_p("Language", styles["table_header"]), _p("Successful cells", styles["table_header_center"]), _p("Gradable cells", styles["table_header_center"]), _p("Task-balanced rate", styles["table_header_center"]), _p("95% task-cluster interval", styles["table_header_center"])]]
    for language in LANGUAGES:
        interval = intervals[language]
        main_rows.append([
            _p(LANGUAGE_LABELS[language], styles["table"]),
            _p(counts[language]["successes"], styles["table"]),
            _p(counts[language]["cells"], styles["table"]),
            _p(_pct(outcomes[language]), styles["table"]),
            _p(f"[{_pct(interval['lower'])}, {_pct(interval['upper'])}]", styles["table"]),
        ])
    story.append(_styled_table(main_rows, [1.3 * inch, 1.15 * inch, 1.05 * inch, 1.35 * inch, 2.05 * inch]))
    story.append(Spacer(1, 0.13 * inch))
    story.append(_p("Paired outcomes", styles["h2"]))
    story.append(_p("Across the 72 task-repetition groups, 9 were all-pass across all three languages and 63 were not all-pass. The current binary outcome pattern contains no language-specific divergence: the same three tasks passed in every language and every repetition.", styles["body"]))
    paired_rows = [[_p("Paired outcome", styles["table_header"]), _p("Groups", styles["table_header_center"])]]
    for key, value in report["paired_outcomes"].items():
        paired_rows.append([_p(key.replace("_", " "), styles["table"]), _p(value, styles["table"])])
    story.append(_styled_table(paired_rows, [3.8 * inch, 1.0 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_p("The three uniformly successful tasks were 001-file, 005-email-triage, and 020-archive-checksum. All other selected tasks failed the strict success threshold in all three language conditions, although many received partial oracle scores.", styles["body"]))

    story.append(PageBreak())
    story.append(_p("3. Efficiency and interaction metrics", styles["h1"]))
    story.append(Image(str(figure_paths["metrics"]), width=6.95 * inch, height=5.35 * inch))
    story.extend(_figure_caption("2", "Operational profile", "Bars show per-cell means. These are descriptive comparisons, not independent significance tests.", styles))
    metric_labels = [
        ("initial_prompt_tokens", "Initial-prompt tokens"),
        ("total_tokens", "Total tokens"),
        ("agent_time_seconds", "Agent time (s)"),
        ("end_to_end_time_seconds", "End-to-end time (s)"),
        ("tool_calls", "Tool calls"),
        ("failed_tool_calls", "Failed tool calls"),
        ("recovery_after_error", "Recovery after error"),
    ]
    metric_rows = [[_p("Metric", styles["table_header"])] + [_p(LANGUAGE_LABELS[language], styles["table_header_center"]) for language in LANGUAGES]]
    for key, label in metric_labels:
        values = report["secondary_metrics"].get(key, {})
        cells = []
        for language in LANGUAGES:
            value = values.get(language)
            cells.append(_pct(value) if key == "recovery_after_error" else _fmt(value, 1))
        metric_rows.append([_p(label, styles["table"])] + [_p(value, styles["table"]) for value in cells])
    story.append(_styled_table(metric_rows, [2.7 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(_p("Interpretation", styles["h2"]))
    story.append(_p("Hindi used more initial-prompt and total tokens on average than English, while Hinglish used fewer total tokens and slightly less time. Hindi also had the highest average failed-tool count and recovery rate. These differences describe this model's observed interaction path; they should not be treated as language-level causal effects without reviewed traces and a larger or replicated design.", styles["body"]))

    story.append(PageBreak())
    story.append(_p("4. Task and category results", styles["h1"]))
    story.append(Image(str(figure_paths["heatmap"]), width=6.65 * inch, height=8.25 * inch))
    story.extend(_figure_caption("3", "Task-level success pattern", "Each cell is the proportion of three repetitions that reached the strict threshold. White separators mark category changes.", styles))

    story.append(PageBreak())
    story.append(_p("Category summary", styles["h2"]))
    category_rows = [[_p("Category", styles["table_header"]), _p("Tasks", styles["table_header_center"]), _p("English", styles["table_header_center"]), _p("Hindi", styles["table_header_center"]), _p("Hinglish", styles["table_header_center"])]]
    for item in data["category_summary"]:
        category_rows.append([
            _p(CATEGORY_LABELS.get(item["category"], item["category"]), styles["table"]),
            _p(item["tasks"], styles["table"]),
            _p(_pct(item["english_rate"]), styles["table"]),
            _p(_pct(item["hindi_rate"]), styles["table"]),
            _p(_pct(item["hinglish_rate"]), styles["table"]),
        ])
    story.append(_styled_table(category_rows, [2.55 * inch, 0.55 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_p("Task-level appendix", styles["h2"]))
    task_rows = [[_p("Task", styles["table_header"]), _p("Category", styles["table_header"]), _p("EN", styles["table_header_center"]), _p("HI", styles["table_header_center"]), _p("Hinglish", styles["table_header_center"]), _p("Oracle score range", styles["table_header_center"])]]
    for item in data["task_summary"]:
        all_scores = [row["oracle_score"] for row in rows if row["task_id"] == item["task_id"] and row["oracle_score"] is not None]
        score_range = "-"
        if all_scores:
            score_range = f"{min(all_scores):.2f}-{max(all_scores):.2f}"
        task_rows.append([
            _p(item["task_id"], styles["table"]),
            _p(CATEGORY_LABELS.get(item["category"], item["category"]), styles["table"]),
            _p(_pct(item["english_rate"]), styles["table"]),
            _p(_pct(item["hindi_rate"]), styles["table"]),
            _p(_pct(item["hinglish_rate"]), styles["table"]),
            _p(score_range, styles["table"]),
        ])
    story.append(_styled_table(task_rows, [1.55 * inch, 1.7 * inch, 0.58 * inch, 0.58 * inch, 0.78 * inch, 1.1 * inch]))
    story.append(_p("Oracle score ranges are shown only where the upstream oracle returned an explicit numeric outcome score; judge-only tasks are shown as '-'.", styles["small"]))

    story.append(PageBreak())
    story.append(_p("5. Limitations and research interpretation", styles["h1"]))
    limitations = [
        ("Translation status", "Hindi and Hinglish were authored before execution, but the team has not yet approved them. A translation defect could affect both model behavior and the interpretation of language comparisons."),
        ("Task sample", "This is a fixed 24-task selection from Harness-Bench, not a random sample of all possible office, shell, retrieval, or software tasks. Task-balanced estimates describe this selection."),
        ("Model and agent", "Only one university-GPU model and one ReAct agent were tested. The result cannot be generalized to other models, agents, serving parameters, or prompting policies."),
        ("Repetitions", "Three repetitions per cell improve observability but are not a substitute for broad random seeds or independent model snapshots. Temperature was fixed at zero."),
        ("Failure attribution", "The traces record observable stages, tool results, and recovery behavior. Claims about whether the model, translation, tool, fixture, or oracle caused a failure should wait for human annotation."),
        ("Privacy and reproducibility", "The report omits the private endpoint, bearer key, and server-backed model identifier. The exact model manifest, raw SQLite database, traces, and workspace artifacts remain local and ignored; the tracked protocol records how to reproduce them with private runtime access."),
    ]
    limitation_rows = [[_p("Limitation", styles["table_header"]), _p("Implication", styles["table_header"])]]
    for title, text in limitations:
        limitation_rows.append([_p(title, styles["table"]), _p(text, styles["table"])])
    story.append(_styled_table(limitation_rows, [1.45 * inch, 5.45 * inch]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(_p("Recommended next research steps", styles["h2"]))
    next_steps = [
        "Review every Hindi and Hinglish overlay against the protected-entity checklist; freeze an approved dataset version without rewriting historical runs.",
        "Annotate a representative sample of traces by failure stage before making causal claims about translation or model behavior.",
        "Replicate the language comparison with additional models or agents only after the Phase I protocol and review workflow are stable.",
    ]
    for index, item in enumerate(next_steps, 1):
        story.append(_p(f"{index}. {_safe(item)}", styles["body"]))

    story.append(PageBreak())
    story.append(_p("Appendix: reproducibility and artifacts", styles["h1"]))
    story.append(_p("The tracked implementation is on branch prishiv_dev. The main study was run as experiment phase1_language_comparison. Raw results and private runtime metadata are intentionally ignored by Git; a local run-index.csv maps each stable cell_id to its trace and result JSON.", styles["body"]))
    artifact_rows = [
        [_p("Artifact", styles["table_header"]), _p("Purpose", styles["table_header"])],
        [_p("docs/phase1_protocol.md", styles["table"]), _p("Secret-free study contract, preflight, execution, review, and completion criteria.", styles["table"])],
        [_p("benchmark/task_selection.yaml", styles["table"]), _p("Pinned source revision, exact 24-task selection, quotas, and source hashes.", styles["table"])],
        [_p("benchmark/translations/phase1.yaml", styles["table"]), _p("English originals plus Hindi and Hinglish overlays with review metadata.", styles["table"])],
        [_p("data/phase1/experiment/provisional_report.json", styles["table"]), _p("Machine-readable findings generated from the main SQLite database; local and ignored.", styles["table"])],
        [_p("data/phase1/experiment/run-index.csv", styles["table"]), _p("Secret-free local index for all 216 cell logs; local and ignored.", styles["table"])],
        [_p("data/phase1/experiment/traces/*.jsonl", styles["table"]), _p("Per-cell event logs including tool arguments/results and usage metadata; local and ignored.", styles["table"])],
    ]
    story.append(_styled_table(artifact_rows, [2.55 * inch, 4.35 * inch]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(_p("This report was generated from the completed database after the final preflight and test audit. Findings should be cited as provisional until translation review and publication decisions are complete.", styles["small"]))
    return story


def _on_page(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(_hex(PALETTE["line"]))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, height - 0.42 * inch, width - document.rightMargin, height - 0.42 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_hex(PALETTE["muted"]))
    canvas.drawString(document.leftMargin, 0.32 * inch, "Phase I findings | provisional | indic-harness-bench")
    canvas.drawRightString(width - document.rightMargin, 0.32 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_pdf(root: Path, database: Path, report_path: Path, output: Path) -> None:
    data = _read_data(root, database, report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="phase1-findings-", dir=root / "tmp" / "pdfs"))
    try:
        figure_paths = {
            "success": temporary / "success.png",
            "metrics": temporary / "metrics.png",
            "heatmap": temporary / "heatmap.png",
        }
        _save_success_chart(figure_paths["success"], data["report"])
        _save_metrics_chart(figure_paths["metrics"], data["report"])
        _save_task_heatmap(figure_paths["heatmap"], data["task_summary"])
        document = SimpleDocTemplate(
            str(output),
            pagesize=letter,
            rightMargin=0.55 * inch,
            leftMargin=0.55 * inch,
            topMargin=0.62 * inch,
            bottomMargin=0.55 * inch,
            title="Phase I Controlled English Hindi Hinglish Benchmark Findings",
            author="indic-harness-bench",
            subject="Provisional Phase I research findings",
        )
        document.build(_build_story(root, data, figure_paths), onFirstPage=_on_page, onLaterPages=_on_page)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase I findings PDF")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--database", type=Path, default=Path("data/phase1/experiment/runs.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("data/phase1/experiment/provisional_report.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/phase1_findings.pdf"))
    args = parser.parse_args()
    root = args.root.resolve()
    database = (root / args.database).resolve() if not args.database.is_absolute() else args.database
    report = (root / args.report).resolve() if not args.report.is_absolute() else args.report
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    build_pdf(root, database, report, output)
    print(output)


if __name__ == "__main__":
    main()
