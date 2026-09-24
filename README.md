# Indic-Harness-Bench

Indic-Harness-Bench is a corrective Phase I benchmark for measuring how
instruction language interacts with executable agent harnesses.

The corrected study compares English, Hindi, and Latin-script Hinglish across
ReAct, NanoBot, and OpenClaw while keeping task resources, model, budgets, and
grading fixed. The earlier 216-cell language-only result is retained as
superseded data and is never pooled with the corrected experiment.

## Repository layout

```text
indic-harness-bench/
├── agents/
├── analysis/
├── benchmark/
│   ├── schemas/
│   └── tasks/
├── configs/
├── data/
├── docker/
├── docs/
├── runner/
├── scripts/
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11+
- Docker Engine/Desktop for isolated runs
- The university OpenAI-compatible endpoint configured in the ignored
  `.env.uni-gpu.local` for Phase I
- Docker Desktop/Linux engine for isolated runs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Phase I execution

The source checkout `../harness-bench` is read-only reference input. The v12
pilot is superseded because Windows checkout conversion changed oracle-sensitive
fixture bytes. The v13 smoke is retained as diagnostic data but superseded after
an audit found that OpenClaw streaming responses omitted usage. The v14 pilot
uses a separate ignored database, requests streamed usage, and excludes streamed
assistant/reasoning text from traces. The 648-cell main study is deferred. See
[the v14 protocol](docs/phase1_protocol_v14.md) for the frozen gates.

```powershell
py -3.14 -m scripts.prepare_phase1 --source ..\harness-bench --selection benchmark/task_selection.v13.yaml --translations benchmark/translations/phase1.v13.yaml --destination data/phase1/tasks-v13 --check-only
docker build -f docker/proxy.Dockerfile -t indic-harness-proxy:pilot-v14 .
py -3.14 -m scripts.provision_phase1_runtime --config configs/phase1.corrected.pilot-v14.yaml --skip-build
py -3.14 -m scripts.calibrate_pilot_budget --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v14.yaml --max-cells 9
# Inspect the nine 001-file grades and traces before proceeding.
py -3.14 -m runner.cli run --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli judge --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli corrected-report --config configs/phase1.corrected.pilot-v14.yaml --output data/phase1/corrected/pilot-v14/provisional_report.md
```

Raw databases, traces, model manifests, and provisional reports live under
ignored `data/phase1/` storage. Each corrected version uses its own database
and experiment ID; v12 and v13 records are never pooled with v14.

## Harness-Bench import

The current Qihoo360 Harness-Bench task layout uses task-local `task.yaml`, prompts, `fixtures/`, and `oracle_grade.py`. The importer copies selected source assets under an Indic-owned `source/` directory, then creates an Indic-owned `task.yaml` containing the three language overlays. It never modifies the reference checkout.

```bash
python -m scripts.import_harness_bench \\
    --source /path/to/harness-bench \\
    --task 001-file \\
    --destination benchmark/tasks/001-file \\
    --translations benchmark/translations/001-file.yaml
```

The translations file must provide `english`, `hindi`, and `hinglish` fields. Import rejects variants that omit protected paths, filenames, or backticked technical entities from the original prompt. Review translations before using their results as research evidence.

## Agents

### ReAct

`agents/react.py` implements a tool-calling loop with the OpenAI Python client against any OpenAI-compatible endpoint. The loop alternates model decisions, tool execution, and observations. Private chain-of-thought is not collected.

### NanoBot

The corrected configuration requires a pinned `indic-harness-nanobot:phase1-pinned`
container image and immutable `image_digest` in `configs/agents.phase1.yaml`.
The preflight intentionally fails if the image or internal network is absent; it
does not fall back to a host process.

### OpenClaw

The corrected configuration likewise requires a pinned
`indic-harness-openclaw:phase1-pinned` image, frozen `image_digest`, and the
Docker-internal model-proxy network. It runs against the fresh workspace with
an isolated model-proxy route.

The old host-process wrappers remain for compatibility but are not valid for
the corrected study.

## Evaluation

Local tasks provide a deterministic command, for example:

```bash
python -m pytest -q grader/test_task.py
```

The run is successful only when the configured expected exit code is returned. Imported Harness-Bench tasks instead execute their copied `oracle_grade.py` and pass when its `outcome_score` reaches the task's configured threshold. Grader/oracle output is stored with the run record.

## Metrics

The corrected primary metric is task-balanced continuous oracle outcome and
paired Hindi/Hinglish deltas against English within each harness. Perfect
completion is secondary. The paper-style completion × process × security score,
tokens, time, tool calls, and error-conditioned recovery are reported
separately; process scoring uses the same university model and is therefore
diagnostic rather than independent.

See `analysis/metrics.py` and `docs/experiment.md`.

## Important limitation

This repository does not vendor the official 106-task Harness-Bench release. The upstream task suite is imported or mounted explicitly so task provenance is visible and benchmark files are not silently modified. The bundled demo task is only a software smoke test and must not be presented as Harness-Bench research data.

## Research references

See `docs/references.md` for the Harness-Bench, ReAct, SWE-bench, SWE-agent, AgentBench, OSWorld, NanoBot, and OpenClaw sources used to shape the implementation.
