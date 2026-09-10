# Indic-Harness-Bench

Indic-Harness-Bench is a Phase 1 benchmark for measuring how instruction language affects executable agent workflows.

Phase 1 compares English, Hindi, and Hinglish instructions while keeping the task, environment, tools, model, agent configuration, execution budget, and grader fixed for the language-only experiment.

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
- An OpenAI-compatible endpoint for the ReAct adapter
- Optional: NanoBot
- Optional: OpenClaw

The default model configuration points to Ollama. Change `configs/models.yaml` before running a real model.

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

## Smoke test

The repository includes a local demo task. It validates task loading, tool safety, grading, storage, and the run path without requiring a research benchmark task.

For local runner development, change `sandbox.mode` to `local` in `configs/phase1.yaml`.

```bash
python -m runner.cli list-tasks
python -m runner.cli run --config configs/phase1.yaml --tasks demo
```

A real Docker run requires the configured container image to be available and the local model endpoint to be reachable.

## Harness-Bench import

The current Qihoo360 Harness-Bench task layout uses task-local `task.yaml`, prompts, `fixtures/`, and `oracle_grade.py`. The importer copies a selected task from a local pinned checkout and does not rewrite upstream task files.

```bash
python -m scripts.import_harness_bench \\
    --source /path/to/harness-bench \\
    --task 001-file \\
    --destination benchmark/tasks/hb_001_file
```

After importing, create the three language variants in the task's `task.yaml` or through the translation preparation process documented in `docs/experiment.md`.

## Agents

### ReAct

`agents/react.py` implements a tool-calling loop with the OpenAI Python client against any OpenAI-compatible endpoint. The loop alternates model decisions, tool execution, and observations. Private chain-of-thought is not collected.

### NanoBot

`agents/external.py` supports native external runtimes. The supplied NanoBot configuration uses its documented one-shot form, `nanobot agent -m ...`.

### OpenClaw

The supplied OpenClaw configuration uses the documented one-shot command surface with `openclaw agent --local ...`.

The external adapters are intentionally disabled in the default Phase 1 configuration. The language-only study should start with one fixed agent and one fixed model.

## Evaluation

Each task provides a deterministic command, for example:

```bash
python -m pytest -q grader/test_task.py
```

The run is successful only when the configured expected exit code is returned. The grader's stdout/stderr are stored with the run record.

## Metrics

The primary metric is success rate by language. Secondary metrics include execution time, token usage when the backend reports it, tool calls, failed tool calls, and run-level recovery behavior.

See `analysis/metrics.py` and `docs/experiment.md`.

## Important limitation

This repository does not vendor the official 106-task Harness-Bench release. The upstream task suite is imported or mounted explicitly so task provenance is visible and benchmark files are not silently modified. The bundled demo task is only a software smoke test and must not be presented as Harness-Bench research data.

## Research references

See `docs/references.md` for the Harness-Bench, ReAct, SWE-bench, SWE-agent, AgentBench, OSWorld, NanoBot, and OpenClaw sources used to shape the implementation.
