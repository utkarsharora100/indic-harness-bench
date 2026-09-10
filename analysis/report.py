from __future__ import annotations

from pathlib import Path

from analysis.metrics import language_deltas, load_runs, success_rates


def build_report(database: Path, output: Path) -> None:
    rows = load_runs(database)
    output.parent.mkdir(parents=True, exist_ok=True)
    rates = success_rates(rows)
    lines = [
        "# Phase 1 Results",
        "",
        "| Language | Success rate |",
        "|---|---:|",
    ]
    for language, rate in sorted(rates.items()):
        lines.append(f"| {language} | {rate:.4f} |")
    lines.extend(["", "## Language deltas", "", "```text", repr(language_deltas(rows)), "```"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
