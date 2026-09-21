from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


LANGUAGES = ("english", "hindi", "hinglish")
LABELS = {"english": "English", "hindi": "Hindi", "hinglish": "Hinglish"}
SERIES = {"english": "--viz-series-1", "hindi": "--viz-series-2", "hinglish": "--viz-series-3"}


def _number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "missing"
    return f"{float(value):,.{digits}f}"


def _percent(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value) * 100:.1f}%"


def _bar_row(language: str, report: dict[str, Any]) -> str:
    rate = report["task_balanced_success"].get(language)
    interval = report["bootstrap_95_task_cluster"].get(language) or {}
    rate_value = float(rate or 0.0)
    low = float(interval.get("lower", 0.0))
    high = float(interval.get("upper", 0.0))
    label = LABELS[language]
    return f"""
      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track" role="img" aria-label="{label}: {_percent(rate)} success; 95 percent interval {_percent(low)} to {_percent(high)}">
          <span class="bar-interval" style="left:{low * 100:.3f}%;width:{max(0.0, (high - low) * 100):.3f}%;"></span>
          <span class="bar-value" style="width:{rate_value * 100:.3f}%;background:var({SERIES[language]});"></span>
        </div>
        <div class="bar-number tabular-nums">{_percent(rate)} <span class="text-muted">[{_percent(low)}, {_percent(high)}]</span></div>
      </div>"""


def _metric_table(report: dict[str, Any]) -> str:
    metrics = report.get("secondary_metrics", {})
    definitions = (
        ("Initial-prompt tokens", "initial_prompt_tokens", 0),
        ("Total tokens", "total_tokens", 0),
        ("Agent time (s)", "agent_time_seconds", 1),
        ("End-to-end time (s)", "end_to_end_time_seconds", 1),
        ("Tool calls", "tool_calls", 1),
        ("Failed tool calls", "failed_tool_calls", 1),
        ("Recovery after error", "recovery_after_error", 1),
    )
    rows = []
    for label, key, digits in definitions:
        values = metrics.get(key, {})
        cells = []
        for language in LANGUAGES:
            value = values.get(language)
            cells.append(_percent(value) if key == "recovery_after_error" else _number(value, digits))
        rows.append(
            "<tr><th scope=\"row\">"
            + html.escape(label)
            + "</th>"
            + "".join(f"<td class=\"tabular-nums\">{html.escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    return "\n".join(rows)


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cells = report.get("cells", {})
    paired = report.get("paired_outcomes", {})
    total = cells.get("gradable_completed", 0)
    successes = {
        language: round(float(report.get("task_balanced_success", {}).get(language, 0.0)) * 24 * 3)
        for language in LANGUAGES
    }
    fragment = f"""<div id="phase1-results-visual" class="phase1-results">
  <style>
    #phase1-results-visual {{
      color: var(--foreground);
      font-family: inherit;
      font-size: var(--font-size-base, 14px);
      line-height: 1.45;
      max-width: 1024px;
    }}
    #phase1-results-visual h2 {{ margin: 0 0 0.35rem; font-size: 1.15rem; font-weight: 500; }}
    #phase1-results-visual p {{ margin: 0 0 1rem; color: var(--muted-foreground); }}
    #phase1-results-visual .plot {{ margin: 1rem 0 1.25rem; }}
    #phase1-results-visual .bar-row {{
      display: grid;
      grid-template-columns: 5.5rem minmax(10rem, 1fr) minmax(11rem, 15rem);
      gap: 0.75rem;
      align-items: center;
      margin: 0.75rem 0;
    }}
    #phase1-results-visual .bar-label {{ font-weight: 500; }}
    #phase1-results-visual .bar-track {{
      height: 1.1rem;
      position: relative;
      background: color-mix(in srgb, var(--muted) 72%, transparent);
      overflow: visible;
    }}
    #phase1-results-visual .bar-value {{ display: block; height: 100%; position: relative; z-index: 2; }}
    #phase1-results-visual .bar-interval {{
      position: absolute;
      top: -0.25rem;
      height: 1.6rem;
      min-width: 2px;
      background: var(--foreground);
      opacity: 0.32;
      z-index: 1;
    }}
    #phase1-results-visual .bar-number {{ white-space: nowrap; }}
    #phase1-results-visual .ticks {{
      display: flex;
      justify-content: space-between;
      margin: 0 0 0 6.25rem;
      color: var(--muted-foreground);
      font-size: 0.8rem;
    }}
    #phase1-results-visual .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem 1.25rem;
      margin: 0.75rem 0 0 6.25rem;
      color: var(--muted-foreground);
      font-size: 0.85rem;
    }}
    #phase1-results-visual .swatch {{ display: inline-block; width: 0.7rem; height: 0.7rem; margin-right: 0.35rem; vertical-align: -0.05rem; background: var(--swatch); }}
    #phase1-results-visual .text-muted {{ color: var(--muted-foreground); font-size: 0.88em; }}
    #phase1-results-visual .tabular-nums {{ font-variant-numeric: tabular-nums; }}
    #phase1-results-visual table {{ width: 100%; border-collapse: collapse; margin-top: 0.7rem; }}
    #phase1-results-visual th, #phase1-results-visual td {{ text-align: right; padding: 0.45rem 0.55rem; border-bottom: 1px solid var(--border); }}
    #phase1-results-visual th:first-child, #phase1-results-visual td:first-child {{ text-align: left; }}
    #phase1-results-visual thead th {{ color: var(--muted-foreground); font-weight: 500; }}
    #phase1-results-visual .summary {{ color: var(--muted-foreground); font-size: 0.9rem; }}
    @media (max-width: 620px) {{
      #phase1-results-visual .bar-row {{ grid-template-columns: 4.5rem minmax(7rem, 1fr); gap: 0.5rem; }}
      #phase1-results-visual .bar-number {{ grid-column: 2; font-size: 0.9rem; }}
      #phase1-results-visual .ticks, #phase1-results-visual .legend {{ margin-left: 5rem; }}
      #phase1-results-visual th, #phase1-results-visual td {{ padding: 0.35rem 0.25rem; }}
    }}
  </style>
  <h2>Phase I provisional results</h2>
  <p class="summary">{total} gradable cells; {successes['english']} English, {successes['hindi']} Hindi, and {successes['hinglish']} Hinglish successes at the strict task threshold. Translation review remains outstanding.</p>
  <div class="plot" role="img" aria-label="Task-balanced success rate by instruction language with 95 percent task-cluster intervals">
    <div class="ticks" aria-hidden="true"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
    {_bar_row("english", report)}
    {_bar_row("hindi", report)}
    {_bar_row("hinglish", report)}
    <div class="legend" aria-label="Chart legend">
      <span><i class="swatch" style="--swatch:var(--viz-series-1)"></i>English</span>
      <span><i class="swatch" style="--swatch:var(--viz-series-2)"></i>Hindi</span>
      <span><i class="swatch" style="--swatch:var(--viz-series-3)"></i>Hinglish</span>
      <span>bar = estimate; line = 95% task-cluster interval</span>
    </div>
  </div>
  <table class="table" aria-label="Phase I secondary metrics by language">
    <caption>Per-cell averages; missing usage is shown as missing, not zero.</caption>
    <thead><tr><th scope="col">Metric</th><th scope="col">English</th><th scope="col">Hindi</th><th scope="col">Hinglish</th></tr></thead>
    <tbody>{_metric_table(report)}</tbody>
  </table>
  <p class="summary">Paired task-repetition outcomes: {paired.get('all_pass', 0)} all-pass groups and {paired.get('mixed', 0)} mixed groups.</p>
</div>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Phase I provisional result visualization fragment")
    parser.add_argument("--report", type=Path, default=Path("data/phase1/experiment/provisional_report.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.report, args.output)


if __name__ == "__main__":
    main()
