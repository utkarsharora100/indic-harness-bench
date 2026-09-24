from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#18324B")
BLUE = colors.HexColor("#2878A5")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#E9A23B")
RED = colors.HexColor("#C65C5C")
PALE = colors.HexColor("#EEF3F7")
INK = colors.HexColor("#263544")
GRAY = colors.HexColor("#647586")
HARNESS_COLORS = {"react": "#2878A5", "nanobot": "#2A9D8F", "openclaw": "#E9A23B"}
PLOT_BLUE = "#2878A5"
PLOT_TEAL = "#2A9D8F"
PLOT_INK = "#263544"
PLOT_NAVY = "#18324B"
LANGUAGES = ("english", "hindi", "hinglish")
HARNESSES = ("react", "nanobot", "openclaw")


def _safe(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "NA" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def _write_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    conditions = [f"{harness}:{language}" for harness in HARNESSES for language in LANGUAGES]
    tasks = sorted({row["task_id"] for row in rows})
    matrix = np.full((len(tasks), len(conditions)), np.nan)
    for i, task in enumerate(tasks):
        for j, condition in enumerate(conditions):
            harness, language = condition.split(":")
            found = next(
                (
                    row["outcome_score"]
                    for row in rows
                    if row["task_id"] == task
                    and row["harness"] == harness
                    and row["language"] == language
                ),
                None,
            )
            if found is not None:
                matrix[i, j] = float(found)
    fig, ax = plt.subplots(figsize=(12.5, 4.7), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conditions)), [x.replace(":", "\n") for x in conditions], fontsize=8)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
    ax.set_title(
        "Continuous LLM outcome score by task, harness, and language", loc="left", weight="bold"
    )
    ax.set_xlabel("Harness : prompt language")
    for i in range(len(tasks)):
        for j in range(len(conditions)):
            label = "NA" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="#18324B")
    fig.colorbar(image, ax=ax, label="Weighted outcome score (0-1)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_paired_plot(rows: list[dict[str, Any]], path: Path) -> None:
    tasks = sorted({row["task_id"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True, constrained_layout=True)
    for ax, harness in zip(axes, HARNESSES, strict=True):
        for task in tasks:
            by_language = {
                row["language"]: row["outcome_score"]
                for row in rows
                if row["task_id"] == task and row["harness"] == harness
            }
            english = by_language.get("english")
            if english is None:
                continue
            for language, offset in (("hindi", -0.06), ("hinglish", 0.06)):
                value = by_language.get(language)
                if value is None:
                    continue
                ax.plot(
                    [0 + offset, 1 + offset],
                    [english, value],
                    marker="o",
                    linewidth=1.4,
                    alpha=0.8,
                    color=PLOT_BLUE if language == "hindi" else PLOT_TEAL,
                )
        ax.set_xticks([0, 1], ["English", "Hindi / Hinglish"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(harness.title(), weight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("LLM outcome score")
    fig.suptitle("Paired task outcomes: English to Indic-language prompts", weight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_delta_plot(report: dict[str, Any], path: Path) -> None:
    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in report["rows"]:
        if row["outcome_score"] is not None and row["status"] == "completed":
            scores[(row["task_id"], row["harness"])][row["language"]] = float(row["outcome_score"])
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    markers = {"hindi": "o", "hinglish": "s"}
    offsets = {"react": -0.18, "nanobot": 0.0, "openclaw": 0.18}
    for harness in HARNESSES:
        for language in ("hindi", "hinglish"):
            xs, ys = [], []
            for task_index, task in enumerate(sorted({key[0] for key in scores})):
                value = scores.get((task, harness), {})
                if "english" in value and language in value:
                    xs.append(
                        task_index + offsets[harness] + (-0.04 if language == "hindi" else 0.04)
                    )
                    ys.append(value[language] - value["english"])
            ax.scatter(
                xs,
                ys,
                label=f"{harness} / {language}",
                marker=markers[language],
                color=HARNESS_COLORS[harness],
                alpha=0.88,
                s=42,
            )
    ax.axhline(0, color=PLOT_INK, linewidth=1, linestyle="--")
    ax.set_xticks(
        range(len(sorted({key[0] for key in scores}))),
        sorted({key[0] for key in scores}),
        rotation=18,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Paired score difference (Indic - English)")
    ax.set_title("Per-task paired language differences", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_oracle_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    for harness in HARNESSES:
        chosen = [
            row
            for row in rows
            if row["harness"] == harness
            and row["status"] == "completed"
            and row["outcome_score"] is not None
            and row["oracle_score"] is not None
        ]
        ax.scatter(
            [row["oracle_score"] for row in chosen],
            [row["outcome_score"] for row in chosen],
            label=harness.title(),
            color=HARNESS_COLORS[harness],
            alpha=0.8,
            s=45,
        )
    ax.plot([0, 1], [0, 1], color=PLOT_INK, linestyle="--", linewidth=1, label="Equal scores")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Original deterministic oracle score")
    ax.set_ylabel("Retrospective LLM outcome score")
    ax.set_title("LLM outcome compared with the original oracle", loc="left", weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_criterion_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    order = [f"{harness}:{language}" for harness in HARNESSES for language in LANGUAGES]
    values = []
    labels = []
    for task in sorted({row["task_id"] for row in rows}):
        criterion_ids = sorted(
            {
                criterion
                for row in rows
                if row["task_id"] == task
                for criterion in json.loads(row["ratings_json"] or "{}").keys()
            }
        )
        for criterion in criterion_ids:
            labels.append(f"{task} / {criterion}")
            line = []
            for condition in order:
                harness, language = condition.split(":")
                match = next(
                    (
                        json.loads(row["ratings_json"])[criterion]["level"]
                        for row in rows
                        if row["task_id"] == task
                        and row["harness"] == harness
                        and row["language"] == language
                        and row["ratings_json"]
                        and criterion in json.loads(row["ratings_json"])
                    ),
                    np.nan,
                )
                line.append(float(match) / 4 if match is not None else np.nan)
            values.append(line)
    if not values:
        return
    matrix = np.asarray(values)
    height = max(5.0, len(labels) * 0.31)
    fig, ax = plt.subplots(figsize=(12.5, height), constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [value.replace(":", "\n") for value in order], fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("Median criterion rating by task and condition", loc="left", weight="bold")
    fig.colorbar(image, ax=ax, label="Criterion level / 4")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_score_spread(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    positions, labels, values, colors_by_harness = [], [], [], []
    position = 0
    for harness in HARNESSES:
        for language in LANGUAGES:
            data = [
                row["outcome_score"]
                for row in rows
                if row["harness"] == harness
                and row["language"] == language
                and row["status"] == "completed"
                and row["outcome_score"] is not None
            ]
            positions.append(position)
            labels.append(f"{harness}\n{language}")
            values.append(data)
            colors_by_harness.append(HARNESS_COLORS[harness])
            position += 1
    box = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.62, showmeans=True)
    for patch, color in zip(box["boxes"], colors_by_harness, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median_line in box["medians"]:
        median_line.set_color(PLOT_NAVY)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("LLM outcome score")
    ax.set_title("Outcome score distributions by harness and language", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.23)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_brief_bars(rows: list[dict[str, Any]], path: Path) -> list[list[str]]:
    measures = (
        ("outcome_score", "LLM outcome (judge-unvalidated)"),
        ("oracle_score", "Original deterministic oracle"),
    )
    language_labels = ("English", "Hindi", "Hinglish")
    colors_by_language = (PLOT_BLUE, PLOT_TEAL, "#E9A23B")
    summaries: dict[tuple[str, str, str], tuple[float | None, int]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.35), sharey=True, constrained_layout=True)
    x = np.arange(len(HARNESSES))
    width = 0.23
    for ax, (field, title) in zip(axes, measures, strict=True):
        for language_index, language in enumerate(LANGUAGES):
            means = []
            counts = []
            for harness in HARNESSES:
                task_values: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    value = row.get(field)
                    if (
                        row["harness"] == harness
                        and row["language"] == language
                        and value is not None
                    ):
                        task_values[row["task_id"]].append(float(value))
                per_task = [float(np.mean(values)) for values in task_values.values() if values]
                mean = float(np.mean(per_task)) if per_task else None
                means.append(mean if mean is not None else 0.0)
                counts.append(len(per_task))
                summaries[(harness, language, field)] = (mean, len(per_task))
            positions = x + (language_index - 1) * width
            bars = ax.bar(
                positions,
                means,
                width,
                label=language_labels[language_index],
                color=colors_by_language[language_index],
            )
            for bar, mean, count in zip(
                bars, (summaries[(h, language, field)][0] for h in HARNESSES), counts, strict=True
            ):
                label = "NA" if mean is None else f"{mean:.2f}\n(n={count})"
                ax.annotate(
                    label,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xticks(x, [h.title() for h in HARNESSES])
        ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Task-balanced mean score (0-1)")
    axes[1].legend(frameon=False, loc="lower right", ncol=1)
    fig.suptitle("Pilot performance by harness and prompt language", weight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    table = [["Harness", "Measure", "English", "Hindi", "Hinglish"]]
    for harness in HARNESSES:
        for field, title in measures:
            values = []
            for language in LANGUAGES:
                mean, count = summaries[(harness, language, field)]
                values.append("NA" if mean is None else f"{mean:.3f} (n={count})")
            table.append([harness.title(), title, *values])
    return table


def _style_table(table: Table, *, header: bool = True, font_size: int = 8) -> None:
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D7E0E8")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("REPEATROWS", (0, 0), (-1, 0)),
        ]
    if header:
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]))
    table.setStyle(TableStyle(commands))


def _page_decor(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.36 * inch, width, 0.36 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(
        0.55 * inch, height - 0.24 * inch, "INDIC HARNESS BENCH  /  PHASE I OUTCOME REJUDGMENT"
    )
    canvas.setStrokeColor(colors.HexColor("#D7E0E8"))
    canvas.line(0.55 * inch, 0.42 * inch, width - 0.55 * inch, 0.42 * inch)
    canvas.setFillColor(GRAY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.55 * inch, 0.24 * inch, "Provisional, retrospective, self-judged pilot")
    canvas.drawRightString(width - 0.55 * inch, 0.24 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_outcome_pdf_v2(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.parent.parent.parent
    scratch = root / "tmp" / "pdfs"
    scratch.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleV2",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=13,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubV2",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=GRAY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadV2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=9,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyV2",
            parent=styles["BodyText"],
            fontSize=9.1,
            leading=13,
            textColor=INK,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallV2", parent=styles["BodyText"], fontSize=7, leading=9, textColor=INK
        )
    )
    styles.add(
        ParagraphStyle(
            name="CalloutV2",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=PALE,
            borderColor=BLUE,
            borderWidth=0.7,
            borderPadding=8,
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    pagesize = landscape(A4)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=pagesize,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.57 * inch,
        bottomMargin=0.57 * inch,
        title=f"Phase I {report['outcome_version']} - Provisional Findings",
        author="Indic Harness Bench research project",
        subject="Retrospective LLM outcome rejudgment of the v14 pilot",
    )
    flow: list[Any] = []
    temporary = tempfile.TemporaryDirectory(prefix="ihb-outcome-v2-", dir=scratch)
    temp = Path(temporary.name)
    flow += [
        Spacer(1, 0.35 * inch),
        Paragraph("PHASE I  /  PILOT REJUDGMENT", styles["SubV2"]),
        Paragraph("Outcome findings, re-scored with an LLM rubric", styles["TitleV2"]),
        Paragraph(f"{report['outcome_version']}  |  Provisional report", styles["SubV2"]),
        Paragraph(
            "This report retrospectively scores the saved v14 agent workspaces with a frozen five-level rubric. "
            "The study contains five purposively selected tasks, three languages, three harnesses, and one run per condition.",
            styles["BodyV2"],
        ),
        Paragraph(
            "CAUTION: the same pinned university model family generated the agent work and judged the outcomes. "
            "Hindi and Hinglish have not received independent bilingual review. "
            f"Calibration status: {report['calibration']['status']}; "
            f"{report['calibration']['checks_failed']} substantive check(s) failed. "
            "Do not treat language differences as validated effects.",
            styles["CalloutV2"],
        ),
    ]
    stats = report["counts"]
    card_data = [
        [
            Paragraph(
                f"<b>{stats['completed_outcome_scores']}</b><br/>numeric outcomes", styles["BodyV2"]
            ),
            Paragraph(f"<b>{stats['needs_review']}</b><br/>review-needed", styles["BodyV2"]),
            Paragraph("<b>1</b><br/>known missing artifact", styles["BodyV2"]),
            Paragraph("<b>29</b><br/>LLM calibration controls", styles["BodyV2"]),
        ]
    ]
    cards = Table(card_data, colWidths=[2.55 * inch] * 4, rowHeights=[0.58 * inch])
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D7E0E8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    flow += [cards, Spacer(1, 9), Paragraph("How to read the scores", styles["HeadV2"])]
    flow.append(
        Paragraph(
            "The primary score is the weighted average of criterion levels divided by four, from 0 to 1. "
            "The archived deterministic oracle score remains a separate measure. The process rating and security gate also remain separate. "
            "Their product with the new LLM outcome is a diagnostic only.",
            styles["BodyV2"],
        )
    )
    flow.append(
        Paragraph(
            "A score is missing when the judge requires human review, a judge call fails, or no agent workspace was produced. "
            "The NanoBot/Hinglish task-050 cell is the one known missing agent artifact. No missing score is imputed.",
            styles["BodyV2"],
        )
    )
    flow.append(PageBreak())

    brief_table_data = _write_brief_bars(report["rows"], temp / "brief-bars.png")
    flow += [
        Paragraph("At a glance: scores by harness and language", styles["HeadV2"]),
        Paragraph(
            "Bars show task-balanced means; labels give the number of task cells contributing to each mean. "
            "The LLM outcome is retrospective and failed substantive calibration checks. Treat it as descriptive only.",
            styles["BodyV2"],
        ),
        Image(str(temp / "brief-bars.png"), width=9.9 * inch, height=3.05 * inch),
    ]
    brief_header_style = ParagraphStyle(
        "BriefHeaderV2", parent=styles["SmallV2"], textColor=colors.white, fontName="Helvetica-Bold"
    )
    brief_table = Table(
        [
            [
                Paragraph(str(value), brief_header_style if row_index == 0 else styles["SmallV2"])
                for value in row
            ]
            for row_index, row in enumerate(brief_table_data)
        ],
        colWidths=[1.1 * inch, 2.5 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch],
        repeatRows=1,
    )
    _style_table(brief_table, font_size=7)
    flow += [
        brief_table,
        Spacer(1, 7),
        Paragraph(
            "Oracle scores are the original executable task grades and are shown separately, not combined with the LLM scores. "
            "NA means no valid score; sample sizes vary by condition.",
            styles["SmallV2"],
        ),
        PageBreak(),
    ]

    flow += [Paragraph("Calibration and method", styles["HeadV2"])]
    cal = report["calibration"]
    calibration_rows = [["Gate", "Result", "Details"]]
    calibration_rows.append(
        [
            "Upstream deterministic oracle controls",
            "PASS"
            if all(item["passed"] for item in cal["deterministic_oracle_checks"])
            else "FAIL",
            "29 synthetic workspaces graded in isolated Docker",
        ]
    )
    calibration_rows.append(
        [
            "LLM ranking, injection, and language controls",
            "PASS" if cal["checks_failed"] == 0 else "FAIL",
            f"{cal['checks_passed']} passed; {cal['checks_failed']} failed out of {len(cal['checks'])} checks",
        ]
    )
    smoke_calls = cal["transport_smoke"]["calls"]
    calibration_rows.append(
        [
            "Proxy transport smoke",
            "PASS",
            f"{smoke_calls} call(s); public alias and token usage verified",
        ]
    )
    calibration_rows.append(
        [
            "Task-001 oracle coverage",
            "Binary",
            "Exact line-count check ignores input preservation; partial control ties the correct control on this oracle.",
        ]
    )
    calibration_table = Table(
        [[_paragraph(str(value), styles["SmallV2"]) for value in row] for row in calibration_rows],
        colWidths=[2.35 * inch, 0.85 * inch, 6.95 * inch],
        repeatRows=1,
    )
    _style_table(calibration_table, font_size=8)
    flow += [
        calibration_table,
        Spacer(1, 10),
    ]
    failed_controls = [item for item in cal["checks"] if not item.get("passed")]
    if failed_controls:
        flow.append(Paragraph("Failed judge controls - findings are unvalidated", styles["HeadV2"]))
        failure_rows = [["Task", "Control", "Observed"]]
        for item in failed_controls:
            observed = item.get("difference", item.get("scores", item))
            failure_rows.append([item.get("task_id", "-"), item.get("check", "-"), str(observed)])
        failure_table = Table(
            [[_paragraph(str(value), styles["SmallV2"]) for value in row] for row in failure_rows],
            colWidths=[2.3 * inch, 3.5 * inch, 4.35 * inch],
            repeatRows=1,
        )
        _style_table(failure_table, font_size=7)
        flow += [failure_table, Spacer(1, 10)]
    flow.append(
        Paragraph("Differences from the original Harness-Bench evaluation", styles["HeadV2"])
    )
    methodology = [
        ["Aspect", "This report", "Original Harness-Bench paper"],
        ["Task coverage", "5 selected tasks; 45 planned cells", "106 benchmark tasks"],
        ["Harnesses", "ReAct, NanoBot, OpenClaw", "Six configurable harnesses"],
        ["Models", "One pinned university model", "Eight model backends"],
        [
            "Outcome scoring",
            "Retrospective LLM rubric for all selected tasks",
            "Executable checks when possible; rubrics otherwise",
        ],
        [
            "Judge independence",
            "Agent model family also judges outcomes",
            "External judge in the paper benchmark",
        ],
        [
            "Research status",
            "One run per cell; exploratory and unreviewed",
            "Benchmark leaderboard study",
        ],
    ]
    method_table = Table(
        [[_paragraph(str(value), styles["SmallV2"]) for value in row] for row in methodology],
        colWidths=[1.65 * inch, 4.05 * inch, 4.45 * inch],
        repeatRows=1,
    )
    _style_table(method_table, font_size=8)
    flow += [
        method_table,
        Spacer(1, 8),
        Paragraph(
            "Numerical leaderboard comparison is not valid because the task set, harnesses, model backends, and scoring procedure differ. "
            'The [Harness-Bench paper](<link href="https://arxiv.org/html/2605.27922">original paper</link>) is the methodological reference.',
            styles["BodyV2"],
        ),
        PageBreak(),
    ]

    mean_data = [["Harness", "English", "Hindi", "Hinglish"]]
    for harness in HARNESSES:
        mean_data.append(
            [
                harness.title(),
                *[
                    _safe(
                        report["task_balanced_harness_language_means"].get(f"{harness}:{language}")
                    )
                    for language in LANGUAGES
                ],
            ]
        )
    flow += [Paragraph("Task-balanced outcomes and paired language contrasts", styles["HeadV2"])]
    mean_table = Table(mean_data, colWidths=[2.2 * inch] * 4)
    _style_table(mean_table, font_size=9)
    flow += [mean_table, Spacer(1, 10)]
    contrast_data = [
        ["Harness", "Contrast", "Paired tasks", "Mean difference", "95% cluster interval"]
    ]
    for harness, contrasts in report["paired_language_deltas"].items():
        for language, difference in contrasts.items():
            interval = report["paired_task_cluster_bootstrap_95"].get(harness, {}).get(language, {})
            n = report["paired_task_counts"].get(harness, {}).get(language, 0)
            contrast_data.append(
                [
                    harness.title(),
                    f"{language.title()} - English",
                    str(n),
                    f"{difference:+.3f}",
                    f"[{interval.get('lower', float('nan')):+.3f}, {interval.get('upper', float('nan')):+.3f}]",
                ]
            )
    contrast_table = Table(
        contrast_data,
        colWidths=[1.55 * inch, 2.0 * inch, 1.25 * inch, 1.65 * inch, 2.3 * inch],
        repeatRows=1,
    )
    _style_table(contrast_table, font_size=8)
    flow += [
        contrast_table,
        Spacer(1, 8),
        Paragraph(
            "Intervals resample the same task clusters across conditions in 20,000 draws (seed 1701). "
            "The task set is small and purposively selected; intervals describe these paired task contrasts and do not establish equivalence or broad population performance.",
            styles["BodyV2"],
        ),
        PageBreak(),
    ]

    _write_heatmap(report["rows"], temp / "heatmap.png")
    _write_paired_plot(report["rows"], temp / "paired.png")
    _write_delta_plot(report, temp / "deltas.png")
    _write_oracle_scatter(report["rows"], temp / "oracle.png")
    _write_criterion_heatmap(report["rows"], temp / "criteria.png")
    _write_score_spread(report["rows"], temp / "spread.png")
    flow += [
        Paragraph("Task-by-condition heatmap", styles["HeadV2"]),
        Image(str(temp / "heatmap.png"), width=9.9 * inch, height=3.7 * inch),
        Spacer(1, 5),
    ]
    flow.append(
        Paragraph(
            "Blank cells represent the known missing agent artifact. Heatmap cells use the primary LLM outcome score.",
            styles["SmallV2"],
        )
    )
    flow.append(PageBreak())
    flow += [
        Paragraph("Paired task outcomes", styles["HeadV2"]),
        Image(str(temp / "paired.png"), width=10.0 * inch, height=3.65 * inch),
    ]
    flow += [
        Spacer(1, 8),
        Image(str(temp / "deltas.png"), width=9.85 * inch, height=3.85 * inch),
        PageBreak(),
    ]
    flow += [
        Paragraph("LLM outcome and original oracle", styles["HeadV2"]),
        Image(str(temp / "oracle.png"), width=6.1 * inch, height=4.9 * inch),
    ]
    flow += [
        Paragraph(
            f"{report['llm_vs_oracle']['comparisons']} cells had both scores. Mean signed LLM-minus-oracle difference: "
            f"{_safe(report['llm_vs_oracle']['mean_signed_difference'])}; mean absolute difference: "
            f"{_safe(report['llm_vs_oracle']['mean_absolute_difference'])}. The measures answer related but different questions.",
            styles["BodyV2"],
        ),
        PageBreak(),
    ]
    flow += [
        Paragraph("Criterion-level ratings", styles["HeadV2"]),
        Image(str(temp / "criteria.png"), width=9.5 * inch, height=5.5 * inch),
        PageBreak(),
    ]
    flow += [
        Paragraph("Score distributions by harness and language", styles["HeadV2"]),
        Image(str(temp / "spread.png"), width=9.8 * inch, height=4.5 * inch),
    ]
    flow += [
        Paragraph(
            "Boxes summarize the small set of completed cell scores. Individual task values are available in the machine-readable tables.",
            styles["BodyV2"],
        ),
        PageBreak(),
    ]

    flow += [Paragraph("Cell-level index and evidence links", styles["HeadV2"])]
    flow.append(
        Paragraph(
            "Each row links to the saved agent trace and workspace archive. Criterion levels, "
            "code-attached submitted evidence and reference fields, pass spread, and judge token "
            "accounting are in the accompanying CSV files.",
            styles["BodyV2"],
        )
    )
    rows = sorted(report["rows"], key=lambda row: (row["task_id"], row["harness"], row["language"]))
    link_data = [["Task", "Harness", "Language", "Status", "LLM", "Oracle", "Trace", "Workspace"]]
    for row in rows:
        trace = row.get("trace_path")
        archive = row.get("workspace_archive")
        trace_link = (
            f'<link href="{Path(trace).as_uri()}" color="#2878A5">open trace</link>'
            if trace
            else "NA"
        )
        archive_link = (
            f'<link href="{Path(archive).as_uri()}" color="#2878A5">open archive</link>'
            if archive
            else "missing"
        )
        link_data.append(
            [
                row["task_id"],
                row["harness"],
                row["language"],
                row["status"],
                _safe(row["outcome_score"]),
                _safe(row["oracle_score"]),
                _paragraph(trace_link, styles["SmallV2"]),
                _paragraph(archive_link, styles["SmallV2"]),
            ]
        )
    link_table = Table(
        link_data,
        colWidths=[
            2.1 * inch,
            0.85 * inch,
            0.85 * inch,
            1.2 * inch,
            0.6 * inch,
            0.6 * inch,
            1.0 * inch,
            1.1 * inch,
        ],
        repeatRows=1,
    )
    _style_table(link_table, font_size=6)
    flow += [link_table, PageBreak()]

    flow += [Paragraph("Method notes, missingness, and interpretation", styles["HeadV2"])]
    for limitation in report["limitations"]:
        flow.append(Paragraph(f"&bull; {limitation}", styles["BodyV2"]))
    flow.append(
        Paragraph(
            "The paired bootstrap has at most five task clusters per harness-language contrast. "
            "The chart and table should therefore be treated as a transparent summary of this pilot, not as a precise estimate for Indic language performance in general.",
            styles["BodyV2"],
        )
    )
    flow.append(Paragraph("Key quantities", styles["HeadV2"]))
    rows_for_summary = [["Measure", "Definition"]]
    for label, definition in (
        ("LLM outcome", "Criterion levels weighted by the frozen task rubric, divided by four."),
        ("Original oracle", "Continuous score stored by the v14 task-specific executable oracle."),
        ("Process score", "Mean of the v14 process-judge dimensions; shown separately."),
        ("Security score", "v14 binary security gate, shown separately."),
        (
            "Combined diagnostic",
            "LLM outcome x process score x security score; missing components are not imputed.",
        ),
        ("Needs review", "Three judgments span more than 0.20; primary score is missing."),
    ):
        rows_for_summary.append([label, definition])
    summary_table = Table(
        [[_paragraph(str(value), styles["SmallV2"]) for value in row] for row in rows_for_summary],
        colWidths=[1.8 * inch, 7.2 * inch],
        repeatRows=1,
    )
    _style_table(summary_table, font_size=8)
    flow.append(summary_table)
    flow.append(Spacer(1, 8))
    flow.append(
        Paragraph(
            "References: Harness-Bench (https://arxiv.org/html/2605.27922); Zheng et al., Judging LLM-as-a-Judge (https://arxiv.org/abs/2306.05685).",
            styles["SmallV2"],
        )
    )
    try:
        document.build(flow, onFirstPage=_page_decor, onLaterPages=_page_decor)
    finally:
        temporary.cleanup()
    return output_path
