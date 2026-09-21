# Phase I protocol and reproducibility guide

This protocol is the tracked, secret-free description of the controlled English,
Hindi, and Hinglish study. Hindi and Hinglish translations were authored before
execution, but remain `unreviewed` until the team approves them. Results are
therefore provisional.

## Fixed study contract

- Branch: `prishiv_dev`.
- Harness-Bench source: `1025086a446653702b80cfb48babbeec35db6b2c`.
- Selection: 24 tasks in `benchmark/task_selection.yaml`, with its recorded
  source tree hashes and category quotas.
- Conditions: English, Hindi, and Latin-script Hinglish; one ReAct agent; one
  university GPU model; temperature `0`; top-p `1`; maximum 2,048 generated
  tokens per model call; maximum 40 agent steps; three repetitions.
- Matrix: 24 × 3 × 3 = 216 stable cells, shuffled with seed `17` and run
  sequentially to avoid shared-GPU contention.
- Isolation: each cell receives a fresh fixture copy in a pinned Docker image;
  the agent container has no network and cannot mount the source oracle. A
  copied workspace is graded in a separate ephemeral container.

## Private runtime setup

Create the ignored `.env.uni-gpu.local` from the university-provided values.
Never put its key, private URL, or server filesystem-backed model ID in source,
reports, traces, or issue comments. The exact served model identity is written
only to the ignored experiment manifest after `/models` and a tool-call check.
The runner stops if that inventory changes.

Build and gate the local image:

```powershell
docker build -f docker/phase1.Dockerfile -t indic-harness-phase1:2026-09-21 .
py -3.14 -m scripts.prepare_phase1 --source ..\harness-bench
py -3.14 -m scripts.preflight_phase1 --config configs/phase1.pilot.yaml --source ..\harness-bench
```

The `harness-bench` checkout is read-only reference input; the preparation
script copies selected tasks into ignored `data/phase1/tasks/`.

## Pilot and main execution

Run the five-task, three-language pilot first:

```powershell
py -3.14 -m runner.cli run --config configs/phase1.pilot.yaml --resume
py -3.14 -m runner.cli report --database data/phase1/pilot/runs.sqlite --output data/phase1/pilot/provisional_report.md
```

Inspect all 15 task traces and grades. If a translation or environment defect is
found, freeze a new dataset/config version; do not rewrite the meaning of old
runs.

Run the main matrix only after the pilot passes infrastructure checks:

```powershell
py -3.14 -m scripts.preflight_phase1 --config configs/phase1.research.yaml --source ..\harness-bench
py -3.14 -m runner.cli run --config configs/phase1.research.yaml --resume
py -3.14 -m runner.cli report --database data/phase1/experiment/runs.sqlite --output data/phase1/experiment/provisional_report.md
```

The main run is complete only when the database has exactly 216 completed,
gradable cells and zero pending or infrastructure-error cells. A low success
rate is a result; a missing cell, changed model identity, or broken grader is
not.

## Logs, provenance, and review

Ignored experiment storage contains the SQLite database, one JSONL trace and
one result JSON per stable cell, the local run manifest, workspace hashes, and
the provisional report. The SQLite `attempt` table retains endpoint retries;
`event` records tool arguments/results, model-call usage when supplied, failed
tool results, and malformed tool-call errors. Private chain-of-thought is never
collected.

Create a convenient secret-free index of all per-cell logs with:

```powershell
py -3.14 scripts/export_run_index.py --experiment-id phase1_language_comparison
```

This writes the ignored `data/phase1/experiment/run-index.csv`; the `cell_id`
column maps each matrix condition to its JSONL trace and result JSON.

Export the translation checklist for team review:

```powershell
py -3.14 -m scripts.review_translations export --output data/phase1/translation_review_checklist.csv
```

Later approvals update review metadata only and do not regenerate prepared
tasks or alter past run records:

```powershell
py -3.14 -m scripts.review_translations mark --task-id 001-file --language hindi --status approved --reviewer name
```

Use `analysis/phase1.py` and the generated JSON report for task-balanced
success, paired deltas, task-cluster bootstrap intervals, usage, timing, tool
failures, and recovery. Infrastructure-error cells are excluded from model
outcomes, and causal failure-stage claims wait for human annotation.

Build the research-facing PDF from the completed local database with:

```powershell
py -3.14 scripts/build_phase1_pdf.py --output output/pdf/phase1_findings.pdf
```

The PDF is a tracked, privacy-safe summary; raw SQLite, traces, results, and
the exact private model manifest remain ignored local artifacts.
