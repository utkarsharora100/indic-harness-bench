# Phase I pilot protocol — canonical-byte correction (v13)

This is the tracked, secret-free protocol for the **five-task, 45-cell pilot**.
The main 648-cell study is deferred. Do not pool experiments.

## Evidence boundary

The v12 pilot completed 45 oracle-gradable cells, but is **superseded for
language inference**. On Windows, checkout conversion changed bytes in
untouched task fixtures. In `016-code-repair-pytest`, this alone broke the
oracle's expected test-file MD5 in every condition. Six pristine source CSVs
in `050-multitable-join-analysis` likewise failed their oracle SHA-256 checks.
Generation limits, missing task tools, and incomplete proxy traces are separate
possible causes of low perfect completion. The v12 results are preserved in
their ignored database and report; none is copied into v13.

The new experiment is `phase1_corrected_pilot_v13`, with distinct manifest,
prepared-task cache, model/runtime/calibration manifests, database, traces,
workspace archives, and reports. Hindi/Hinglish translations are unchanged and
still unreviewed, so all language findings remain provisional.

## Frozen design

- Work and push only on `prishiv_dev`.
- Pinned Harness-Bench Git commit:
  `1025086a446653702b80cfb48babbeec35db6b2c`. The selection manifest is
  `benchmark/task_selection.v13.yaml`; translation overlay is
  `benchmark/translations/phase1.v13.yaml`. The 24 task definitions and 72
  language variants are unchanged apart from provenance hashes of **raw Git
  blobs**. No checkout line-ending conversion is permitted.
- Pilot tasks: `001-file`, `016-code-repair-pytest`,
  `050-multitable-join-analysis`, `025-meeting-action-tracker`, and
  `019-incident-runbook-synthesis`. Each has English, Hindi, and Hinglish
  prompts and ReAct, native NanoBot, and native OpenClaw runs: 45 fresh cells.
- Primary descriptive outcome: each pinned task's continuous oracle score in
  `[0,1]`. Perfect score is secondary, not the headline measure. Also retain
  complete oracle checks, model/agent stop reasons, tool errors, token usage,
  elapsed time, per-cell JSONL trace, and final workspace archive.
- Task/harness/language order is shuffled within blocks using seed `1701` and
  run sequentially. Temperature `0` does not make repetitions independent
  samples. This pilot has only **one** execution per condition and five
  purposively selected tasks; uncertainty intervals are exploratory only.
- Same university model, prompt variant, fixtures, oracle, external task
  timeout, at most 40 model calls, temperature `0`, and top-p `1`. ReAct and
  native harnesses have different intrinsic tools/recovery behavior by design.
  All three images must expose the same Python/pandas/pytest and base CLI task
  tools. The generation cap is selected **before** cells run by synthetic
  probes at 4,096 then 8,192 tokens, and frozen in the config. A failed probe
  blocks the pilot; changing the cap after any run requires another version.
- Grading uses an isolated copy of the workspace. The oracle and reference
  answers never enter agent containers. Agents reach only a per-cell model
  proxy through a Docker-internal network; the proxy alone has university
  credentials and outbound access. The same model judges process/security
  **afterward**, making those scores diagnostic rather than independent proof.

## Reproduction gates

Keep `.env.uni-gpu.local` ignored. It contains the private endpoint URL, key,
and optional requested model. Never paste those values into tracked files,
logs, PDFs, or public issues. The exact served model ID is stored only in an
ignored local manifest. Model identity, proxy alias, image IDs, dataset,
calibration, and rubric versions must match on resume.

From the project root, with the reference checkout at `..\harness-bench`:

```powershell
py -3.14 -m pytest -q
py -3.14 -m scripts.prepare_phase1 --source ..\harness-bench --selection benchmark/task_selection.v13.yaml --translations benchmark/translations/phase1.v13.yaml --destination data/phase1/tasks-v13 --check-only
py -3.14 -m scripts.provision_phase1_runtime --config configs/phase1.corrected.pilot-v13.yaml
py -3.14 -m scripts.calibrate_pilot_budget --config configs/phase1.corrected.pilot-v13.yaml
```

For a new machine, omit `--check-only` on `prepare_phase1` to build the
ignored cache once. The provision script builds the shared task-tool image,
benchmark image, NanoBot v0.3.5 image, OpenClaw v2026.9.5 image, and proxy;
it creates/checks the internal Docker network and records local image IDs.
Inspect the calibration result. If it selects 8,192, change
`generation.max_tokens` from 4,096 to 8,192 in the v13 config **before any
pilot cell**. If neither probe passes, stop and investigate; do not run the
pilot with an untested limit.

```powershell
py -3.14 -m scripts.preflight_phase1 --config configs/phase1.corrected.pilot-v13.yaml --source ..\harness-bench
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v13.yaml --max-cells 9
py -3.14 -m runner.cli judge --config configs/phase1.corrected.pilot-v13.yaml
# Inspect the nine 001-file conditions before resuming the other 36 cells.
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v13.yaml --resume
py -3.14 -m runner.cli judge --config configs/phase1.corrected.pilot-v13.yaml
py -3.14 -m runner.cli corrected-report --config configs/phase1.corrected.pilot-v13.yaml --output data/phase1/corrected/pilot-v13/provisional_report.md
py -3.14 scripts/export_run_index.py --database data/phase1/corrected/pilot-v13/runs.sqlite --output data/phase1/corrected/pilot-v13/run-index.csv --experiment-id phase1_corrected_pilot_v13
```

`runner.cli run` invokes the same preflight before writing cells. The gate
checks 24 pinned source hashes, 72 variants, the pristine `016` and `050`
oracle hashes, all image IDs, shared task tools, native startup, internal-only
network and denied direct internet, proxy tool call and trace, served-model
identity, and frozen calibration. Any failed gate stops execution.

Pilot acceptance requires exactly 45 completed, oracle-gradable cells, no
unresolved infrastructure errors, intact proxy traces/workspace archives, and
45 valid process judgments. Inspect every grade and blinded representative
traces before interpreting scores. A correction to prompts, runtime, or scoring
gets a **new version**; do not edit completed records. The prior 216-cell
experiment and v12 pilot remain readable but are superseded for inference.

The local SQLite, JSONL, archives, provisional report, and PDF remain ignored.
The report's trace index links each cell to its artifacts. Low oracle scores
are possible research results; missing grades, truncated logs, changed model
identity, or invalid judge output are infrastructure failures. Do not claim
language causality or equivalence from this pilot. Translation review is still
required before any definitive Indic-language conclusion.
