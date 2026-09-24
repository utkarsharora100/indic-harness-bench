"""Readable provisional PDF for the outcome-v8 pilot rejudgment."""

from __future__ import annotations

from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
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

from analysis.outcome_pdf_v2 import _write_heatmap, _write_oracle_scatter, _write_paired_plot

HARNESSES = ("react", "nanobot", "openclaw")
LANGUAGES = ("english", "hindi", "hinglish")
NAVY = colors.HexColor("#18324B")
INK = colors.HexColor("#263544")
BLUE = "#2878A5"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"


def _bar_chart(report: dict[str, Any], path: Path) -> None:
    means = report["means"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10.5, 4.3), constrained_layout=True)
    for index, (language, color) in enumerate(zip(LANGUAGES, (BLUE, TEAL, GOLD), strict=True)):
        vals = [means.get(f"{harness}:{language}", 0) for harness in HARNESSES]
        bars = ax.bar(x + (index - 1) * 0.24, vals, width=0.23, label=language.title(), color=color)
        for bar, value in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, [h.title() for h in HARNESSES])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean LLM outcome score (0–1)")
    ax.set_title("Pilot outcome by harness and prompt language", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _score_distribution(rows: list[dict[str, Any]], path: Path) -> None:
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(8.5, 3.4), constrained_layout=True)
    for harness, color in zip(HARNESSES, (BLUE, TEAL, GOLD), strict=True):
        values = [
            row["outcome_score"] for row in rows
            if row["harness"] == harness and row["outcome_score"] is not None
        ]
        ax.hist(
            values, bins=bins, histtype="step", linewidth=2.2,
            label=harness.title(), color=color,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("LLM outcome score")
    ax.set_ylabel("Cell count")
    ax.set_title("Distribution of judged cell scores", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _table(data: list[list[str]], widths: list[float]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D6E0E7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.34 * inch, width, 0.34 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(0.55 * inch, height - 0.23 * inch, "INDIC HARNESS BENCH  /  OUTCOME-V8")
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.55 * inch, 0.25 * inch, "Retrospective, self-judged, provisional")
    canvas.drawRightString(width - 0.55 * inch, 0.25 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_v8_pdf(report: dict[str, Any], path: Path) -> Path:
    """Keep simple performance graphics first; audit and caveats remain prominent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title, head, body, small = (
        styles["Title"],
        styles["Heading2"],
        styles["BodyText"],
        styles["Normal"],
    )
    title.textColor = head.textColor = NAVY
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.55 * inch,
        title="Phase I pilot, outcome-v8: provisional reference-guided rejudgment",
    )
    flow: list[Any] = []
    with TemporaryDirectory(prefix="ihb-v8-charts-") as temporary:
        folder = Path(temporary)
        bars, heat, scatter, paired, distribution = (
            folder / "bars.png",
            folder / "heat.png",
            folder / "scatter.png",
            folder / "paired.png",
            folder / "distribution.png",
        )
        _bar_chart(report, bars)
        _write_heatmap(report["rows"], heat)
        _write_oracle_scatter(report["rows"], scatter)
        _write_paired_plot(report["rows"], paired)
        _score_distribution(report["rows"], distribution)
        flow.extend(
            [
                Spacer(1, 0.17 * inch),
                Paragraph("Phase I pilot: reference-guided LLM rejudgment", title),
                Paragraph(
                    "Outcome-v8 scores saved final workspaces. The same pinned university "
                    "model judged its own agents' output; this is exploratory, not independent.",
                    body,
                ),
                Paragraph(
                    f"Coverage: {report['coverage'].get('completed', 0)} scored, "
                    f"{report['coverage'].get('needs_review', 0)} review-needed, "
                    f"{report['coverage'].get('judge_error', 0)} technical judge errors, "
                    "one agent infrastructure failure out of 45 planned cells. "
                    f"Recorded model-call errors before retry: {report['judge_error_calls']}. "
                    f"Calibration: {escape(report['calibration_status'])}.",
                    body,
                ),
                Image(str(bars), width=8.9 * inch, height=3.65 * inch),
                PageBreak(),
                Paragraph("At-a-glance means and paired language differences", head),
            ]
        )
        summary = [["Harness", "English", "Hindi", "Hinglish", "Hindi–English", "Hinglish–English"]]
        for harness in HARNESSES:
            row = [harness.title()]
            row.extend(
                f"{report['means'].get(f'{harness}:{language}', float('nan')):.3f} "
                f"(n={report['mean_task_counts'].get(f'{harness}:{language}', 0)})"
                for language in LANGUAGES
            )
            row.extend(
                f"{report['paired_differences'].get(harness, {}).get(language, float('nan')):+.3f}"
                for language in ("hindi", "hinglish")
            )
            summary.append(row)
        flow.extend(
            [
                _table(
                    summary,
                    [1.3 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.4 * inch, 1.6 * inch],
                ),
                Spacer(1, 0.14 * inch),
                Paragraph(
                    "Means are descriptive and use the available task scores (n shown above); "
                    "all task-050 outcomes are unresolved. One run was used per condition. "
                    "The NanoBot/Hinglish/task-050 agent artifact is missing, not scored zero. "
                    "Translations and judge fairness await two bilingual human reviewers.",
                    body,
                ),
                Image(str(heat), width=8.9 * inch, height=3.35 * inch),
                PageBreak(),
                Paragraph("Paired task outcomes and oracle audit", head),
                Image(str(paired), width=8.9 * inch, height=3.6 * inch),
                Spacer(1, 0.08 * inch),
                Paragraph(
                    "The deterministic oracle is an audit comparison, not the primary outcome. "
                    "Disagreement can reflect a valid alternative or judge error; inspect work.",
                    body,
                ),
                PageBreak(),
                Paragraph("LLM-versus-oracle agreement and calibration", head),
                Image(str(scatter), width=7.3 * inch, height=3.7 * inch),
                Image(str(distribution), width=7.8 * inch, height=3.0 * inch),
            ]
        )
        control_matrix = [[
            "Task", "Good", "Partial", "Wrong", "Missing", "Near miss", "Alternative"
        ]]
        for task_id in sorted({key.split(":")[0] for key in report["calibration_cases"]}):
            values = report["calibration_cases"]
            row = [task_id.split("-")[0]]
            for kind in (
                "good", "partial", "incorrect", "missing", "near_miss", "valid_alternative"
            ):
                item = values.get(f"{task_id}:{kind}", {})
                score = item.get("score")
                row.append("NEEDS REVIEW" if score is None else f"{score:.2f}")
            control_matrix.append(row)
        failed = report["calibration_failed_checks"]
        flow.append(
            Paragraph(
                f"Calibration: 39 controls; {len(failed)} failed checks. "
                + (
                    "The judge is NOT validated; all scores are descriptive."
                    if failed
                    else "All pre-specified checks passed, but human agreement is still pending."
                ),
                body,
            )
        )
        flow.extend(
            [Spacer(1, 0.06 * inch), _table(
                control_matrix,
                [1.25 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch,
                 0.8 * inch, 0.9 * inch, 0.95 * inch],
            )]
        )
        if failed:
            flow.extend(
                [
                    PageBreak(),
                    Paragraph("Calibration checks that failed", head),
                    Paragraph(
                        "These pre-specified controls failed; the judge is not validated. "
                        "The pilot scores are retained as descriptive outputs, not evidence "
                        "that the judge is accurate.",
                        body,
                    ),
                ]
            )
            for check in failed:
                observed = ", ".join(
                    f"{escape(str(key))}={escape('missing' if value is None else f'{value:.3f}')}"
                    for key, value in check.get("observed", {}).items()
                )
                flow.append(
                    Paragraph(
                        f"<b>{escape(str(check['check']))}</b><br/>Observed: {observed}",
                        body,
                    )
                )
        flow.extend(
            [
                PageBreak(),
                Paragraph("Audit trail and limitations", head),
                Paragraph(
                    "Question, reference, and submission were kept separate. No raw input "
                    "tables, oracle score, agent identity, or language label reached the judge. "
                    "Exact fields require exact values; semantic checks allow equivalent wording. "
                    "Focused verification by the same model is correlated, not independent.",
                    body,
                ),
                Paragraph(
                    "Machine-readable cells.csv and requirements.csv give every rating, "
                    "evidence ID, usage, oracle and v7 scores, trace and workspace paths. "
                    "Original v14 agent results and v7 judgments were not modified.",
                    body,
                ),
                Paragraph(
                    "This is a retrospective measure change on five tasks, not a new agent run. "
                    "One run per condition and a self-judge do not support a validated Indic "
                    "language advantage. Human review remains pending.",
                    body,
                ),
            ]
        )
        flow.extend([PageBreak(), Paragraph("Cell-by-cell trace and workspace index", head)])
        link_rows: list[list[Any]] = []
        for row in sorted(
            report["rows"], key=lambda item: (item["task_id"], item["harness"], item["language"])
        ):
            trace = Path(row["trace"])
            trace_link = Paragraph(
                f'<link href="{escape(trace.as_uri())}" color="#2878A5">open trace</link>',
                small,
            )
            workspace = row["workspace_archive"]
            workspace_link = (
                Paragraph(
                    f'<link href="{escape(Path(workspace).as_uri())}" '
                    'color="#2878A5">open archive</link>', small,
                )
                if workspace else "missing"
            )
            link_rows.append(
                [row["task_id"].split("-")[0], row["harness"], row["language"],
                 "NA" if row["outcome_score"] is None else f'{row["outcome_score"]:.3f}',
                 trace_link, workspace_link]
            )
        header = ["Task", "Harness", "Language", "V8", "Trace", "Workspace"]
        widths = [0.75 * inch, 1.2 * inch, 1.1 * inch, 0.65 * inch, 1.3 * inch, 1.5 * inch]
        for start in range(0, len(link_rows), 12):
            if start:
                flow.extend(
                    [
                        PageBreak(),
                        Paragraph("Trace and workspace index (continued)", head),
                    ]
                )
            flow.append(
                _table(
                    [header, *link_rows[start : start + 12]],
                    widths,
                )
            )
        doc.build(flow, onFirstPage=_page, onLaterPages=_page)
    return path
