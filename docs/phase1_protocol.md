# Phase I corrective protocol and reproducibility guide

This protocol is the tracked, secret-free description of the controlled English,
Hindi, and Hinglish study. Hindi and Hinglish translations were authored before
execution, but remain `unreviewed` until the team approves them. Results are
therefore provisional.

## Study status

The earlier `phase1_language_comparison` 216-cell experiment is retained as a
superseded pilot. It is not pooled with the corrective study because its runner
used an inconsistent `/workspace` file-tool path, omitted upstream task hooks,
truncated oracle output, and reported binary scores at a perfect-score
threshold. The corrected experiment uses a new database and experiment ID.

## Fixed study contract

- Branch: `prishiv_dev`.
- Harness-Bench source: `1025086a446653702b80cfb48babbeec35db6b2c`.
- Selection: 24 tasks in `benchmark/task_selection.yaml`, with its recorded
  source tree hashes and category quotas.
- Conditions: English, Hindi, and Latin-script Hinglish; ReAct, NanoBot, and
  OpenClaw harnesses; one university GPU model; temperature `0`; top-p `1`;
  maximum 2,048 generated tokens per model call; maximum 40 agent steps; three
  fresh repetitions.
- Matrix: 24 × 3 × 3 × 3 = 648 stable cells, with a 45-cell pilot first.
  Conditions are randomized within task/attempt blocks using seed `1701` and
  run sequentially to avoid shared-GPU contention. Temperature-zero repetitions
  are not treated as independent random draws.
- Isolation: each cell receives a fresh fixture copy in a pinned Docker image;
  the source oracle is inaccessible to agents. A copied workspace is graded in
  a separate ephemeral container. Native harness images are pinned separately.
- Scoring: the complete continuous upstream `outcome_score` is primary;
  perfect completion is secondary. The paper-style process/security aggregate
  is recorded as diagnostic and uses the same university model as judge, so it
  is not an independent validation.

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
py -3.14 -m scripts.preflight_phase1 --config configs/phase1.corrected.pilot.yaml --source ..\harness-bench
```

The `harness-bench` checkout is read-only reference input; the preparation
script copies selected tasks into ignored `data/phase1/tasks/`.

Before the pilot, provision the official stable NanoBot and OpenClaw releases
as the image names in `configs/agents.phase1.yaml`, record each immutable
`image_digest`, and create the Docker-internal `phase1-agent-net` network. The
preflight rejects missing or mismatched digests and rejects non-internal
networks. It does not substitute the old host-process wrappers.

For a new Docker engine, the network shape is:

```powershell
docker network create --internal phase1-agent-net
```

The native image entrypoints and their exact stable release versions must be
recorded by the team when those images are provisioned; this repository does
not invent or silently substitute a NanoBot/OpenClaw release.

## Pilot and main execution

Run the corrected 45-cell pilot first:

```powershell
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot.yaml --resume
py -3.14 -m runner.cli judge --config configs/phase1.corrected.pilot.yaml
py -3.14 -m runner.cli corrected-report --config configs/phase1.corrected.pilot.yaml --output data/phase1/corrected/pilot/provisional_report.md
```

Inspect all 15 task traces and grades. If a translation or environment defect is
found, freeze a new dataset/config version; do not rewrite the meaning of old
runs.

Run the main matrix only after the pilot passes infrastructure checks:

```powershell
py -3.14 -m scripts.preflight_phase1 --config configs/phase1.corrected.research.yaml --source ..\harness-bench
py -3.14 -m runner.cli run --config configs/phase1.corrected.research.yaml --resume
py -3.14 -m runner.cli judge --config configs/phase1.corrected.research.yaml
py -3.14 -m runner.cli corrected-report --config configs/phase1.corrected.research.yaml --output data/phase1/corrected/provisional_report.md
```

The main run is complete only when the database has exactly 648 completed,
oracle-gradable cells, zero pending or infrastructure-error cells, and 648
valid process judgments for the paper-style diagnostic tables. A low outcome
score is a result; a missing cell, changed model identity, broken grader, or
invalid judge response is not.

## Logs, provenance, and review

Ignored experiment storage contains the SQLite database, one JSONL trace and
one result JSON per stable cell, the local run manifest, workspace hashes, and
the provisional report. The SQLite `attempt` table retains endpoint retries;
`event` records tool arguments/results, model-call usage when supplied, failed
tool results, and malformed tool-call errors. Private chain-of-thought is never
collected.

Create a convenient secret-free index of all per-cell logs with:

```powershell
py -3.14 scripts/export_run_index.py --database data/phase1/corrected/runs.sqlite --output data/phase1/corrected/run-index.csv --experiment-id phase1_corrected_v1
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

Use `analysis/corrected.py` and the generated JSON report for task-balanced
continuous completion, paired language deltas within each harness, task-cluster
bootstrap intervals, process/security components, usage, timing, tool failures,
and error-conditioned recovery. Infrastructure-error cells and invalid grades
are excluded from model outcomes, and causal failure-stage claims wait for
human annotation.

Build the research-facing PDF from the completed local database with:

```powershell
py -3.14 scripts/build_corrected_pdf.py --report data/phase1/corrected/provisional_report.json --output data/phase1/corrected/provisional_report.pdf
```

The PDF is a tracked, privacy-safe summary; raw SQLite, traces, results, and
the exact private model manifest remain ignored local artifacts.
