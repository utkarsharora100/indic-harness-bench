# Indic-Harness-Bench — Repository Reference

## Repository map

```text
agents/      Agent contracts, tool schemas, ReAct, and external wrappers
analysis/    Result metrics, reports, statistics, and failure taxonomy
benchmark/   Task models, discovery, translation checks, schemas, and task data
configs/     Agent/model and experiment matrices
data/        Generated-result and trace destinations
docker/      Reproducible Python/pytest image
docs/        Architecture, research protocol, plan, and references
runner/      CLI, configuration, sandbox, grading, persistence, orchestration
scripts/     Harness-Bench import command
tests/       Unit and integration tests
```

## Runtime flow

```text
Config + task YAML → shuffled task/language/agent/model/repetition cells
→ fresh workspace → adapter/tool loop → grader or upstream oracle
→ SQLite + JSONL + JSON summary → metrics/report
```

## Root and CI files

### `.gitignore`
**Brief:** Exclusion policy for local/generated files.

**Features:** Ignores caches, environments, builds, environment files, databases, outputs, temp files, and mutable task workspaces; explicitly retains `.gitkeep` and demo source.

**Interfaces / usage:** Used by Git only.

### `LICENSE`
**Brief:** License terms.

**Features / interfaces:** No runtime behavior.

**Usage:** Legal distribution/reuse reference.

### `pyproject.toml`
**Brief:** Package and development-tool configuration.

**Features:** Python >=3.11; runtime dependencies `docker`, `openai`, `pydantic`, `pyyaml`, `rich`, `typer`; dev dependencies pytest/Ruff; Hatch packages; pytest/Ruff settings.

**Interfaces:** Console command `indic-harness` → `runner.cli:app`.

**Usage:** Consumed by pip, Hatch, pytest, Ruff.

### `README.md`
**Brief:** User guide.

**Features:** Installation, test, run, import, agent, grading, metric, and scope documentation.

**Interfaces / usage:** Documents `runner.cli` and `scripts.import_harness_bench` commands.

### `IMPLEMENTATION_NOTES.md`
**Brief:** Delivered-scope and boundary statement.

**Features:** Explains upstream-task provenance, demo limitation, and external-wrapper isolation limitation.

**Interfaces / usage:** Contributor/researcher reference.

### `.github/workflows/test.yml`
**Brief:** GitHub Actions CI workflow.

**Features:** On push/PR: checkout, Python 3.12, install `.[dev]`, pytest, compileall.

**Interfaces / usage:** Workflow `test`; automatic CI.

## `agents`

### `agents/__init__.py`
**Brief:** Agent-package export surface.

**Variables:** `__all__` restricts/documents the intended public imports: `AgentAdapter`, `AgentRequest`, `AgentResponse`, and `ExternalAgentAdapter`.

**Usage:** Enables `agents.*` imports and explicit public-symbol exports.

### `agents/base.py`
**Brief:** Provider-neutral adapter data contract.

**Features:** Separates runner inputs/outputs from concrete agent implementations.

**Interfaces and functions:**

- `AgentRequest(instruction, workspace, system_prompt, model, temperature, top_p, max_tokens, max_steps, command_runner=None)` — run input; `command_runner` delegates shell work to Docker when present.
- `AgentResponse(completed, text, usage={}, metadata={})` — run output; usage stores tokens and metadata stores events/provider details.
- `AgentAdapter` — protocol requiring `name`, `run(request)`, and `close()`.

**Usage:** Created by `ExperimentRunner`, consumed by adapters, and persisted after each run.

### `agents/factory.py`
**Brief:** YAML-driven adapter factory.

**Features:** Supports `react` and `external` types; reads React secret from configured environment-variable name.

**Functions:**

- `build_agent(name, agent_config, model_config, timeout_seconds) -> AgentAdapter` — builds ReAct using endpoint/key/timeout, or external adapter using executable and argument lists; raises for unknown type.

**Usage:** Called by `ExperimentRunner._run_cell`.

### `agents/react.py`
**Brief:** OpenAI-compatible ReAct implementation.

**Features:** Tool-call loop, token aggregation, observable event logging, tool-exception conversion, and maximum-step termination.

**Interfaces and functions:**

- `ReactConfig(base_url, api_key, timeout_seconds)` — connection configuration.
- `ReactAgent.__init__(config)` — creates `OpenAI` client.
- `ReactAgent.run(request) -> AgentResponse` — sends system/user messages and tool schemas, executes requested workspace tools, adds results back to messages, returns final text or max-step response.
- `ReactAgent.close()` — closes client.
- `build_react_agent(base_url, api_key, timeout_seconds) -> AgentAdapter` — factory helper.

**Usage:** Returned for `type: react`; covered by `tests/test_react_trace.py`.

### `agents/tools.py`
**Brief:** Workspace tools exposed to ReAct.

**Features:** Workspace traversal protection, UTF-8 file I/O, recursive text search, local/Docker command execution, timeout/output handling.

**Interfaces and functions:**

- `TOOLS` — OpenAI schemas for `list_files`, `read_file`, `write_file`, `search_files`, `run_command`.
- `WorkspaceTools(workspace, timeout_seconds, command_runner=None)` — establishes resolved workspace boundary.
- `schemas()` — returns `TOOLS`.
- `execute(name, arguments)` — dispatches a valid tool or raises `ValueError`.
- `_resolve(relative_path)` — rejects paths outside workspace.
- `_list_files(path)` — returns sorted direct entries/types.
- `_read_file(path)` — reads UTF-8 file or returns structured error.
- `_write_file(path, content)` — creates parents and writes UTF-8.
- `_search_files(query, path)` — lists UTF-8 text files containing query.
- `_run_command(command)` — uses Docker callback or local shell; captures result/timeout.
- `serialise_tool_result(result)` — JSON-encodes tool result for model messages.

**Usage:** Instantiated per ReAct run; covered by `tests/test_tools.py`.

### `agents/external.py`
**Brief:** One-shot CLI adapter for external agent runtimes.

**Features:** Direct/file instruction transport, subprocess lifecycle, timeout/missing-binary handling, OpenClaw final JSON parsing.

**Interfaces and functions:**

- `ExternalAgentConfig(command, args, message_args, message_file_args, timeout_seconds)` — external runtime contract.
- `ExternalAgentAdapter(name, config)` — adapter instance.
- `run(request) -> AgentResponse` — writes `.benchmark_message.txt`, invokes command in workspace, builds process response.
- `close()` — no-op.
- `_parse_openclaw_json(stdout)` — parses final nonempty JSON line or returns parse-error flag.

**Usage:** Built for `type: external` configs.

### `agents/gemini.py`
**Brief:** Commented future Gemini adapter sketch.

**Features / interfaces:** No active symbols.

**Usage:** Reference only.

## `benchmark`

### `benchmark/__init__.py`, `benchmark/schemas/__init__.py`, `benchmark/tasks/__init__.py`
**Brief:** Empty package markers. **Usage:** Enable package imports.

### `benchmark/models.py`
**Brief:** Pydantic task contract.

**Features:** Rejects unknown fields, enforces required language variants/positive limits/nonempty ID, and requires a local judge or upstream oracle.

**Interfaces and functions:**

- `Language` — `english | hindi | hinglish` literal type.
- `Instructions(english, hindi, hinglish)` — translated instructions.
- `Environment(image, workspace, fixtures=None)` — workspace settings.
- `Limits(max_steps, timeout_seconds)` — positive execution limits.
- `Judge(command, workdir=".", expected_exit_code=0)` — local grading settings.
- `UpstreamTask(source_dir="source", prompt_file="prompt.txt", fixtures_dir="fixtures", oracle_module="oracle_grade.py", expected_outcome_score=1.0)` — imported-task oracle settings.
- `TaskDefinition(...)` — complete validated task.
- `task_id_not_empty(value)` — validator.
- `requires_a_grader()` — validator requiring `judge` or `upstream`.
- `instruction_for(language)` — returns selected variant.
- `from_file(path)` — YAML load and model validation.

**Usage:** Loader, runner, importer, and task tests.

### `benchmark/loader.py`
**Brief:** Task discovery/selection.

**Functions:**

- `discover_tasks(root)` — sorted-loads immediate `*/task.yaml` files.
- `select_tasks(root, task_ids)` — resolves selected IDs; `["demo"]` selects `source_task: local_demo`; errors on unknown IDs.

**Usage:** CLI list/run; covered by `tests/test_loader.py`.

### `benchmark/translation.py`
**Brief:** Translation technical-entity guard.

**Features:** Detects backticked values, paths, and common technical filenames.

**Interfaces and functions:**

- `TECHNICAL_PATTERNS` — regex constant tuple.
- `protected_entities(text)` — extracts normalized unique protected entities.
- `missing_entities(source, variant)` — returns entities omitted from a variant.

**Usage:** Import validation; covered by `tests/test_translation.py`.

### `benchmark/upstream.py`
**Brief:** Compatibility layer for copied Harness-Bench assets.

**Functions:**

- `copy_fixtures(fixtures, workspace)` — creates `in/`/`out/` and overlays fixtures.
- `render_instruction(template, workspace)` — replaces `$WORKSPACE`.
- `run_oracle(task_dir, oracle_module, workspace)` — dynamically loads `score_workspace`; returns score-zero error data on failures.

**Usage:** Sandbox setup, prompt construction, and upstream grading.

### `benchmark/schemas/task.schema.yaml`
**Brief:** Human-readable task field/type outline.

**Features:** Documents task, instruction, environment, limits, and judge fields.

**Interfaces / usage:** Author reference; runtime validation is `TaskDefinition`.

### `benchmark/schemas/trace.schema.json`
**Brief:** JSON Schema for trace events.

**Features:** Requires run ID, nonnegative step, event type, timestamp, and data object.

**Interfaces / usage:** Documents JSONL output from runner.

### `benchmark/task_selection.yaml`
**Brief:** Research selection manifest.

**Features:** Upstream repository/revision, target count, category quotas, empty task list.

**Interfaces / usage:** Manual planning; not read by current runner.

### `benchmark/translations/001-file.yaml`
**Brief:** English/Hindi/Hinglish overlays for imported `001-file`.

**Features:** Preserves `$WORKSPACE`, input/output paths, and constraints.

**Interfaces / usage:** Import input.

### `benchmark/tasks/README.md`
**Brief:** Task-directory conventions. **Usage:** Task-author reference.

### `benchmark/tasks/demo/task.yaml`
**Brief:** Local smoke-test definition.

**Features:** Multilingual addition instruction, `workspace`, 10-step/60-second limits, pytest judge.

**Interfaces / usage:** `TaskDefinition`; default CLI demo via `source_task: local_demo`.

### `benchmark/tasks/demo/workspace/app.py`
**Brief:** Expected completed demo source.

**Functions:** `add(a, b)` — returns the sum `a + b`.

**Usage:** Imported by demo grader; target for agent edits.

### `benchmark/tasks/demo/grader/test_task.py`
**Brief:** Demo deterministic grader.

**Functions:** `test_add()` — checks two addition cases.

**Usage:** Invoked by demo `judge.command`.

### `benchmark/tasks/demo/fixtures/.gitkeep`
**Brief:** Keeps empty fixture directory tracked. **Usage:** Git marker only.

### `benchmark/tasks/001-file/task.yaml`
**Brief:** Indic-owned imported task definition.

**Features:** Three overlays, provenance, 20-step/600-second limits, upstream oracle configuration.

**Interfaces / usage:** `TaskDefinition`; used by pilot/integration test.

### `benchmark/tasks/001-file/source/task.yaml`
**Brief:** Copied upstream metadata.

**Features:** Original title/class/asset names/timeout/tags.

**Interfaces / usage:** Importer source/provenance.

### `benchmark/tasks/001-file/source/prompt.txt`
**Brief:** Canonical English line-count instruction.

**Features:** Requires line count in `out/linecount.txt`, preserves input.

**Interfaces / usage:** `$WORKSPACE` source; translation comparison source.

### `benchmark/tasks/001-file/source/fixtures/in/input.txt`
**Brief:** Four-line task input (`alpha` through `delta`).

**Usage:** Copied into fresh upstream workspace.

### `benchmark/tasks/001-file/source/oracle_grade.py`
**Brief:** Upstream deterministic grade function.

**Functions:** `score_workspace(workspace)` — checks stripped `out/linecount.txt` equals `"4"`, returns detailed check and 0.0/1.0 score.

**Usage:** Loaded by `run_oracle`.

### `benchmark/tasks/001-file/source/llm_rubric.py`
**Brief:** Preserved unused upstream rubric bridge.

**Functions / interfaces:** `_defaults()` loads relative upstream default rubric; globals `RUBRIC_SYSTEM` and `USER_TEMPLATE` receive its values.

**Usage:** Asset preservation only; current runner uses oracle.

## `runner`

### `runner/__init__.py`
**Brief:** Empty package marker. **Usage:** Enables `runner.*` imports.

### `runner/config.py`
**Brief:** Experiment-YAML wrapper.

**Interfaces and functions:**

- `ExperimentConfig(path, data)` — configuration dataclass.
- `root` — repository root calculated from config path.
- `load(path)` — resolved YAML loader.
- `experiment`, `generation`, `sandbox`, `storage` — section properties.

**Usage:** CLI and runner configuration.

### `runner/sandbox.py`
**Brief:** Fresh workspace context manager.

**Features:** Copies source, overlays fixtures, supports local/Docker mode, removes containers/temp files on exit.

**Interfaces and functions:**

- `WorkspaceSandbox(source_workspace, image, mode, fixtures=None)` — lifecycle setup.
- `workspace` — active temporary workspace path.
- `__enter__()` — initializes local/Docker environment.
- `run_command(command)` — runs `sh -lc` in Docker and returns tool-style result.
- `__exit__(exc_type, exc, tb)` — cleanup.

**Usage:** Entered once per experiment cell.

### `runner/grader.py`
**Brief:** Common local/upstream grader adapter.

**Interfaces and functions:**

- `GradeResult(success, returncode, stdout, stderr, elapsed_seconds)` — unified result.
- `run_grader(workspace, command, workdir, expected_exit_code, timeout_seconds, command_runner=None)` — runs judge locally or Docker callback.
- `run_upstream_oracle(task_dir, oracle_module, workspace, expected_outcome_score)` — applies oracle score threshold.

**Usage:** Called by runner; result stored as grade record.

### `runner/log.py`
**Brief:** SQLite persistence layer.

**Features:** Tasks, language variants, runs, events, grades, and manual failure annotations; JSON fields preserve Unicode.

**Interfaces and functions:**

- `SCHEMA` — SQLite DDL.
- `RunStore(path)` — opens DB/applies schema.
- `close()` / `commit()` — lifecycle.
- `add_task(...)`, `add_language_variant(...)` — provenance/instruction upserts.
- `start_run(...)` — initial run record.
- `add_event(...)` — trace-event insertion.
- `add_grade(...)` — named grade upsert.
- `finish_run(run_id, **values)` — final aggregates/metadata update.

**Usage:** Owned by `ExperimentRunner`; queried by analysis.

### `runner/runner.py`
**Brief:** Core experiment executor.

**Features:** Seeded matrix shuffle, task/variant persistence, clean sandbox, adapter lifecycle, trace/artifact writes, oracle/judge invocation, finally-block finalization.

**Interfaces and functions:**

- `SYSTEM_PROMPT` — fixed workspace safety/completion prompt.
- `ExperimentRunner(config, agents, models)` — creates store/output dirs.
- `close()` — commits and closes store.
- `run(tasks)` — forms and shuffles task × language × agent × model × repetition cells.
- `_run_cell(task, language, agent_name, model_name, repetition, seed)` — executes/persists one complete cell.
- `_event(run_id, step, event_type, data)` — standard UTC trace event.

**Usage:** CLI run command; integration-tested upstream task path.

### `runner/cli.py`
**Brief:** Typer command-line interface.

**Interfaces and functions:**

- `app` — Typer application; console entry target.
- `console` — Rich console.
- `load_yaml(path)` — agents/models YAML loader.
- `list_tasks(tasks_dir=...)` — `list-tasks` command.
- `run(config=..., tasks="demo")` — `run` command; comma-separated tasks.
- `report(database=..., output=...)` — `report` command.

**Usage:** `python -m runner.cli` or installed `indic-harness`.

## `analysis`

### `analysis/__init__.py`
**Brief:** Empty package marker. **Usage:** Enables imports.

### `analysis/metrics.py`
**Brief:** Stored-run language metrics.

**Functions:**

- `load_runs(database)` — reads all SQLite `run` rows as dicts.
- `success_rates(rows)` — success fraction grouped by language.
- `language_deltas(rows)` — Hindi−English and Hinglish−English deltas.

**Usage:** Report generation; `tests/test_metrics.py`.

### `analysis/report.py`
**Brief:** Markdown report writer.

**Functions:** `build_report(database, output)` — writes sorted success table and delta block.

**Usage:** CLI `report`.

### `analysis/statistics.py`
**Brief:** Proportion uncertainty utility.

**Functions:** `proportion_interval(successes, total, z=1.96)` — validates inputs and returns bounded Wilson interval.

**Usage:** Analysis extension point; `tests/test_statistics.py`.

### `analysis/failure_analysis.py`
**Brief:** Failure taxonomy and paired-outcome helper.

**Interfaces and functions:**

- `FAILURE_CATEGORIES` — allowed taxonomy set.
- `validate_category(category)` — validates a category or raises `ValueError`.
- `paired_outcome(english_success, hindi_success, hinglish_success)` — assigns all-pass, language-specific, both-Indic, English-only, or mixed label.

**Usage:** Manual analysis; `tests/test_failure_analysis.py`.

## Config, Docker, and scripts

### `configs/agents.yaml`
**Brief:** Named adapter configuration.

**Features:** React, NanoBot direct message, and OpenClaw message-file command definitions.

**Interfaces / usage:** `agents.<name>` dictionaries passed to factory.

### `configs/models.yaml`
**Brief:** Named model endpoint configuration.

**Features:** Local Ollama-compatible Qwen model and API-key environment name; commented Gemini plan.

**Interfaces / usage:** `models.<name>` dictionaries used by factory/runner.

### `configs/phase1.yaml`
**Brief:** Default local demo experiment.

**Features:** 3 repetitions, 3 languages, ReAct/local model, deterministic generation, local sandbox, default storage.

**Interfaces / usage:** Default CLI config.

### `configs/pilot.yaml`
**Brief:** Imported-task pilot experiment.

**Features:** One repetition of `001-file` per language and separate pilot database.

**Interfaces / usage:** Manual end-to-end check.

### `configs/phase1.research.yaml`
**Brief:** Docker full-study template.

**Features:** Empty task list pending pinned selection; 3 repetitions and Docker mode.

**Interfaces / usage:** Research-study starting point.

### `configs/README.md`
**Brief:** Fairness rule for configurations.

**Features / usage:** States only instruction language should vary in language-only work.

### `docker/Dockerfile`
**Brief:** Python 3.12 Slim benchmark image.

**Features:** Unbuffered/no-bytecode Python, `/workspace`, current pip, pytest, idle command.

**Interfaces / usage:** Image configured by `sandbox.image`.

### `docker/compose.yaml`
**Brief:** Compose service definition.

**Features / interfaces:** Builds Dockerfile as `benchmark` service with `/workspace` and sleep command.

**Usage:** Local Docker image/service development.

### `scripts/__init__.py`
**Brief:** Empty package marker. **Usage:** Enables imports.

### `scripts/import_harness_bench.py`
**Brief:** Import CLI for a pinned upstream task.

**Features:** Checks upstream layout, translation coverage and protected entities, destination absence; copies source assets; writes Indic-owned task YAML.

**Interfaces and functions:**

- `app` — Typer app.
- `import_task(source, task, destination, translations)` — import command and complete validation/copy/generation workflow.

**Usage:** `python -m scripts.import_harness_bench --source ... --task ... --destination ... --translations ...`.

## Tests

### `tests/test_loader.py`
**Brief:** Loader tests. **Functions:** `test_demo_is_discoverable()`, `test_demo_alias_selects_demo()`. **Usage:** pytest coverage for loader.

### `tests/test_task_model.py`
**Brief:** Task-model test. **Functions:** `test_demo_task_loads()`. **Usage:** validates three instructions.

### `tests/test_translation.py`
**Brief:** Translation guard tests. **Functions:** `test_technical_entities_are_detected()`, `test_missing_entity_is_reported()`.

### `tests/test_tools.py`
**Brief:** Workspace-tool tests. **Functions:** `test_workspace_write()`, `test_workspace_rejects_parent_path()`.

### `tests/test_react_trace.py`
**Brief:** Network-free ReAct trace test.

**Features:** Fake client/response helpers simulate one `read_file` call then final output.

**Functions:** `test_react_records_tool_calls(tmp_path)` asserts completion, counters, and event.

**Usage:** ReAct loop regression coverage.

### `tests/test_metrics.py`
**Brief:** Metric test. **Functions:** `test_language_metrics()`. **Usage:** verifies rates/delta.

### `tests/test_statistics.py`
**Brief:** Statistical sanity test. **Functions:** `test_proportion_interval_contains_point_estimate()`.

### `tests/test_failure_analysis.py`
**Brief:** Taxonomy test. **Functions:** `test_paired_outcomes()`, `test_failure_category_validation()`.

### `tests/test_upstream_task.py`
**Brief:** End-to-end imported-task runner test.

**Interfaces and functions:**

- `LineCountAgent` — test adapter that records instructions and writes expected output.
- `LineCountAgent.__init__()` — creates instruction list.
- `LineCountAgent.run(request)` — writes `out/linecount.txt` with `4` and returns success.
- `LineCountAgent.close()` — no-op.
- `test_runner_executes_copied_upstream_task_in_all_languages(tmp_path, monkeypatch)` — copies task, injects adapter, runs all languages, checks instructions and SQLite grades.

**Usage:** Highest-level runner/oracle/persistence coverage.

## Project docs and runtime directories

### `docs/architecture.md`
**Brief:** Component architecture. **Features:** Boundaries, isolation, trace policy. **Usage:** Technical orientation.

### `docs/experiment.md`
**Brief:** Controlled Phase 1 protocol. **Features:** variable controls, translation/grading rules, metrics, failure approach. **Usage:** Experiment reference.

### `docs/phase1_plan.md`
**Brief:** Detailed research plan. **Features:** rationale, proposed design, metrics, logging, taxonomy, reproducibility, rollout. **Usage:** Strategic reference; some examples are aspirational.

### `docs/references.md`
**Brief:** External source list. **Features:** Benchmark/agent research links. **Usage:** Provenance.

### `data/results/.gitkeep`
**Brief:** Result-directory marker. **Usage:** Runner writes ignored `<run_id>.json` files beside it.

### `data/traces/.gitkeep`
**Brief:** Trace-directory marker. **Usage:** Runner writes ignored `<run_id>.jsonl` files beside it.

## Variable reference

This section supplements the interface entries above. It records state-bearing and decision-bearing variables; conventional loop variables (`row`, `item`, `exc`) are described only when their role is not obvious from the surrounding function.

### `agents/base.py`

**Variables:** `AgentRequest.instruction` is the selected language instruction; `workspace` is the temporary filesystem root; `system_prompt` is fixed benchmark guidance; `model` identifies the backend model; `temperature`, `top_p`, `max_tokens`, and `max_steps` bound generation; `command_runner` optionally sends shell operations into the sandbox. `AgentResponse.completed` marks normal completion; `text` is agent/process output; `usage` holds backend token totals; `metadata` holds events and adapter-specific diagnostics.

### `agents/factory.py`

**Variables:** `kind` chooses adapter branch; `api_key_name` selects the environment variable; `api_key` is its value, defaulting to `ollama`. `name`, `agent_config`, `model_config`, and `timeout_seconds` are factory inputs forwarded to the selected adapter.

### `agents/react.py`

**Variables:** `ReactAgent.name` is constant `react`; `self.config` retains connection settings; `self.client` is the OpenAI client. In `run`, `tools` is the workspace tool executor; `messages` is the evolving API conversation; `input_tokens`/`output_tokens` aggregate usage; `response_text` stores latest assistant text; `events` stores externally observable tool events; `tool_calls` and `failed_tool_calls` are aggregate counters. Per iteration, `completion` is the backend result, `message` its selected choice, `message_tool_calls` its requested calls, `arguments` decoded JSON, and `result` the tool success/error object. `step` is the one-based logical turn reported to traces.

### `agents/tools.py`

**Variables:** `TOOLS` is the model-visible tool schema. `self.workspace` is the resolved safety root; `self.timeout_seconds` bounds local commands; `self.command_runner` is optional Docker delegation. `handlers` maps allowed names to methods; `path`, `target`, and `root` are resolved paths; `content`/`text` are UTF-8 file contents; `matches` is the search-result list; `process` is local subprocess output. Tool results consistently use `ok` plus either data or error fields.

### `agents/external.py`

**Variables:** Config fields `command`, `args`, `message_args`, `message_file_args`, and `timeout_seconds` fully define the subprocess invocation. Adapter `self.name` selects adapter identity and OpenClaw parsing; `self.config` retains those settings. `workspace` is request workspace; `message_file` is `.benchmark_message.txt`; `command` is the final argv list; `started` measures duration; `process` is subprocess result; `metadata` stores return code, stderr, elapsed time, and optional OpenClaw object. In parser, `lines` are nonblank stdout lines and `data` is decoded final JSON.

### `benchmark/models.py`

**Variables:** `Language` restricts selected languages. Model fields are task data: `Instructions.english/hindi/hinglish`; `Environment.image/workspace/fixtures`; `Limits.max_steps/timeout_seconds`; `Judge.command/workdir/expected_exit_code`; and `UpstreamTask.source_dir/prompt_file/fixtures_dir/oracle_module/expected_outcome_score`. `TaskDefinition.model_config` forbids extra YAML fields; its `task_id`, `category`, `source_task`, `instruction`, `environment`, `limits`, `judge`, and `upstream` constitute the validated runtime task. `data` in `from_file` is parsed YAML before validation.

### `benchmark/loader.py` and `benchmark/translation.py`

**Variables:** Loader `tasks` is discovered definitions; `by_id` is ID lookup; `missing` is unresolved requested IDs. Translation `TECHNICAL_PATTERNS` defines extraction rules; `entities` accumulates source identifiers; `entity` is a normalized match. `source` and `variant` in `missing_entities` are canonical and translated instruction text.

### `benchmark/upstream.py`

**Variables:** `fixtures` and `workspace` identify source/destination; `child` is a fixture entry and `destination` its copied target. `template` and `workspace` drive marker rendering. Oracle loading uses `oracle_path`, unique `module_name`, import `spec`, loaded `module`, callable `scorer`, and returned `result`; all error paths carry `outcome_score: 0.0`.

### `benchmark/tasks/001-file/source/oracle_grade.py` and `llm_rubric.py`

**Variables:** Oracle `target` is output path; `value` is its stripped content or empty string; `ok` is exact equality with `"4"`, used in check detail and score. Rubric bridge `g` is default rubric path, `spec` its import specification, `m` the loaded module, and globals `RUBRIC_SYSTEM`/`USER_TEMPLATE` are extracted rubric values.

### `runner/config.py` and `runner/cli.py`

**Variables:** Config `path` is YAML location and `data` its parsed mapping; `root` derives project root. CLI `app` is the Typer surface and `console` is Rich output. In CLI run, `experiment_config` is parsed config; `root` locates companion YAML; `agents` and `models` are their mappings; `task_defs` is selected task list; `runner` executes it; `results` are summaries; `passed` is success count. `table` holds list-task display rows.

### `runner/sandbox.py`

**Variables:** `self.source_workspace` and `self.fixtures` are resolved optional inputs; `self.image`/`self.mode` select runtime; `self.root` is temporary root; `self.client` and `self.container` are Docker resources. `workspace` property is `<root>/workspace`. `result` is Docker exec result; `stdout_text`/`stderr_text` are decoded, truncated outputs. `exc_type`, `exc`, and `tb` are context-manager cleanup inputs.

### `runner/grader.py`

**Variables:** `GradeResult` fields are normalized grade state. `started` measures grade duration; callback/local `result` or `process` contains command outcome. Oracle `score` is result score, `success` is threshold comparison, and `payload` is JSON serialized into grade stdout. `expected_exit_code` and `expected_outcome_score` define success rules.

### `runner/log.py`

**Variables:** `SCHEMA` is all table DDL. `self.connection` is SQLite connection. `path` identifies database. Event `tool`, `arguments`, and `result` are optional observed tool data. `values` in `finish_run` provides final required timing/success/counter fields and optional token/metadata fields.

### `runner/runner.py`

**Variables:** `SYSTEM_PROMPT` is fixed across language conditions. Runner instance fields `self.config`, `self.agents`, `self.models`, and `self.store` hold experiment dependencies; `database` is configured SQLite path. `experiment` is selected experiment mapping; `seed` controls randomized ordering; `cells` is Cartesian run matrix. Per cell, `run_id`, `start_time`, and `started` identify/time execution; `trace` accumulates events; `response` and `grade` begin absent to ensure finalization works after errors. `model_config`, `task_root`, `source_workspace`, and `fixtures` select input environment; `agent`, `workspace_for_prompt`, and `request` form execution input; `trace_event` is persisted event form. Finally, `elapsed`, `usage`, `success`, `trace_path`, and `result_path` produce stored aggregates/artifacts. `_event` fields `run_id`, `step`, `event_type`, `timestamp`, and `data` form the trace contract.

### `analysis/metrics.py`, `report.py`, `statistics.py`, and `failure_analysis.py`

**Variables:** Metrics `connection` is short-lived SQLite access; `totals` and `successes` are language counters; `language` selects a row bucket; `rates` is computed rate mapping; `english` is baseline rate. Report `rows` are loaded runs, `rates` computed rates, and `lines` Markdown output. Statistics `p` is observed proportion, `denominator`, `centre`, and `margin` are Wilson interval terms; `z` is critical value. Failure `FAILURE_CATEGORIES` is allowed-label set; `values` is the English/Hindi/Hinglish boolean tuple.

### `scripts/import_harness_bench.py`

**Variables:** `app` is Typer CLI. Inputs `source`, `task`, `destination`, and `translations` define import source/target. `source_task` is selected upstream folder; `required` and `missing` validate asset layout; `translation_data` is raw YAML; `instruction` is either nested or direct language mapping; `required_languages` and `missing_languages` validate variants; `prompt` is canonical text; `invalid_variants` maps languages to omitted entities; `upstream_definition` is copied metadata; `definition` is generated Indic-owned task YAML.

### Test variables

**Variables:** Test-local `workspace`, `tools`, `result`, `task`, `rows`, `rates`, `source`, and `variant` create isolated inputs/assertion targets. `test_react_trace.py` fake fields (`FakeCall.id`, `function.name`, `function.arguments`, fake message/tool calls, fake usage, and `FakeCompletions.calls`) model an OpenAI interaction. `test_upstream_task.py` uses `LineCountAgent.name`, `instructions`, temporary `root`, `source_task`, `destination_task`, `config`, `agent`, `runner`, and `results` to validate end-to-end behavior.
