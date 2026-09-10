# Implementation notes

## Included

- Task schema and loader
- English/Hindi/Hinglish instruction fields
- Technical-entity validation
- ReAct tool-calling adapter using the OpenAI Python client
- NanoBot and OpenClaw native CLI wrappers
- Fresh per-run workspace creation
- Docker execution mode for shell commands and deterministic graders
- JSONL trace output
- SQLite run, event, language-variant, grade, and failure storage
- Success-rate and language-delta analysis
- Proportion confidence intervals
- Failure taxonomy and paired outcome helpers
- CLI task discovery, execution, and report generation
- Unit tests and GitHub Actions test workflow

## Deliberate boundaries

The official Harness-Bench task data is not copied into this repository. The importer requires a local pinned checkout. This avoids silently changing or redistributing benchmark content and makes the source revision explicit in `benchmark/task_selection.yaml`.

The bundled `demo` task is a runner smoke test, not an evaluation task from Harness-Bench and must not be used as research evidence.

External NanoBot/OpenClaw execution is a host subprocess integration. For a strict container security boundary, use the corresponding upstream Harness-Bench adapters or package the external runtime into the benchmark image.
