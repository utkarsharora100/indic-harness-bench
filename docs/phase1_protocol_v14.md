# Phase I corrective pilot protocol - v14

This is the frozen protocol for the **five-task, 45-cell pilot**. The 648-cell
main study remains deferred. Work is limited to `prishiv_dev`; pilot data,
traces, model metadata, and reports stay in ignored local storage.

## Version boundary

The v12 experiment is superseded because Windows checkout conversion changed
oracle-sensitive fixture bytes. The v13 experiment is also superseded for
inference: a nine-cell `001-file` smoke block completed, but the audit found
that OpenClaw streaming responses did not return usage to the proxy, leaving its
token totals missing. The v13 database, traces, and archives are preserved and
must not be pooled with v14.

The v14 proxy requests the terminal usage chunk on streamed completions and
records token counts in the same fields as ReAct and NanoBot. Streamed trace
summaries retain finish reasons and tool-call deltas but omit assistant text
and reasoning channels. Initial-prompt usage is the first observed prompt-token
count, while total usage is summed across calls. The task set, prompt overlays,
task/runtime images, model provider, grader, and task budgets otherwise remain
fixed. v14 has its own experiment ID, model/runtime/calibration manifests, and
database. The endpoint's physical GPU model was not independently verified;
the A6000 description is unconfirmed.

## Frozen design

- Upstream Harness-Bench commit:
  `1025086a446653702b80cfb48babbeec35db6b2c`.
- Selection and translation files:
  `benchmark/task_selection.v13.yaml` and
  `benchmark/translations/phase1.v13.yaml`; prepared cache:
  `data/phase1/tasks-v13`. The 24 task definitions and 72 language variants
  retain canonical Git-blob bytes. English prompts are canonical; Hindi and
  Hinglish remain unreviewed, so language findings are provisional.
- Pilot subset: `001-file`, `016-code-repair-pytest`,
  `050-multitable-join-analysis`, `025-meeting-action-tracker`, and
  `019-incident-runbook-synthesis`. Conditions are English, Hindi, and
  Hinglish crossed with ReAct, NanoBot, and OpenClaw: 45 cells total, one fresh
  execution per cell.
- The task-balanced continuous oracle score in `[0,1]` is primary and perfect
  completion is secondary. Five purposively selected tasks are not a random
  sample; uncertainty intervals are exploratory and do not establish
  equivalence or population-level language effects.
- Temperature `0`, top-p `1`, 8,192 output tokens per model call, no more than
  40 calls per cell, and each task's external timeout are frozen. The 8,192
  cap passed both pre-run synthetic generation probes; 4,096 truncated the
  longer probe. Native and ReAct tools remain intrinsically different.
- Execution is sequential. Condition ordering is shuffled within each
  task/repetition block with seed `1701`. The single pilot execution per
  condition is not an independent repeated sample.
- All agents use the pinned university model behind the public proxy alias.
  Agents cannot directly reach the public internet; the sidecar alone has
  outbound access and holds the endpoint credentials. The exact served model
  identity is retained only in the ignored local manifest.
- The oracle grades a copy of the workspace in isolation. The upstream source,
  reference answers, and oracle code are inaccessible to the agent.
- Process/security judgments use the same university model after agent runs;
  these are diagnostic, not independent validation. Invalid or missing
  judgments remain missing.

## Required gates

Before cells begin, `runner.cli run` must pass its full gate: pinned source and
translation checks; prepared-task and post-hook fixture hashes; baseline oracle
checks; calibrated cap/model match; frozen image IDs; native startup and
egress-denial checks; shared task-tool parity; proxy call/usage trace; and
secret redaction. The first nine cells are the complete `001-file` smoke block.
Proceed to the remaining 36 only if all nine have completed oracle grades,
usable traces and workspace archives, correct public model routing, and
non-missing usage for all three harnesses.

The full pilot is accepted only when all 45 cells are completed and oracle
gradable, with no unresolved infrastructure errors; traces and workspace
archives are present; usage is retained when the endpoint returns it; and all
45 process judgments validate against the frozen rubric. An oracle exception,
missing grade, proxy/runtime failure, or invalid judgment is not a model score
of zero.

## Reproduction

With the pinned source checkout at `..\harness-bench`, run from the repository
root:

```powershell
py -3.14 -m pytest -q
py -3.14 -m scripts.prepare_phase1 --source ..\harness-bench --selection benchmark/task_selection.v13.yaml --translations benchmark/translations/phase1.v13.yaml --destination data/phase1/tasks-v13 --check-only
docker build -f docker/proxy.Dockerfile -t indic-harness-proxy:pilot-v14 .
py -3.14 -m scripts.provision_phase1_runtime --config configs/phase1.corrected.pilot-v14.yaml --skip-build
py -3.14 -m scripts.calibrate_pilot_budget --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v14.yaml --max-cells 9
```

Inspect the nine smoke cell grades, trace integrity, proxy usage, and workspace
archives before continuing:

```powershell
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli judge --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli corrected-report --config configs/phase1.corrected.pilot-v14.yaml --output data/phase1/corrected/pilot-v14/provisional_report.md
```

Do not change prompts, graders, model identity, task tools, generation limits,
or image versions during v14. A material correction requires a new version and
fresh affected cells. The local provisional report must state that translations
are unreviewed and that this is a small pilot, not the main study.
