from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from agents.base import AgentRequest, AgentResponse
from agents.factory import build_agent
from benchmark.models import Language, TaskDefinition
from benchmark.upstream import render_instruction
from runner.config import ExperimentConfig
from runner.grader import GradeResult, GraderInfrastructureError, run_grader, run_upstream_oracle
from runner.inference import InferenceTransientError, public_model_config
from runner.log import RunStore
from runner.redaction import redact
from runner.sandbox import SandboxInfrastructureError, WorkspaceSandbox


SYSTEM_PROMPT = (
    "You are operating in a controlled benchmark workspace. "
    "Complete the user's task by using the available tools and commands. "
    "Do not access files outside the workspace. Stop when the required workspace state is complete."
)


@dataclass(frozen=True, slots=True)
class Cell:
    task: TaskDefinition
    language: Language
    agent: str
    model: str
    repetition: int
    seed: int
    cell_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def workspace_sha256(workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        relative = path.relative_to(workspace).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def stable_cell_id(
    experiment_id: str,
    dataset_revision: str,
    task_id: str,
    language: str,
    agent: str,
    model: str,
    repetition: int,
) -> str:
    raw = "\x1f".join(
        [experiment_id, dataset_revision, task_id, language, agent, model, str(repetition)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def is_transient_endpoint_error(exc: Exception) -> bool:
    if isinstance(exc, InferenceTransientError):
        return True
    name = type(exc).__name__
    if name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ConflictError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    return status in {408, 425, 429} or (isinstance(status, int) and status >= 500)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, agents: dict, models: dict) -> None:
        self.config = config
        self.agents = agents
        self.models = models
        database = config.root / config.storage["database"]
        self.store = RunStore(database)
        (config.root / config.storage["traces_dir"]).mkdir(parents=True, exist_ok=True)
        (config.root / config.storage["results_dir"]).mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.store.commit()
        self.store.close()

    def plan_cells(self, tasks: list[TaskDefinition]) -> list[Cell]:
        experiment = self.config.experiment
        configured_ids = experiment.get("tasks")
        actual_ids = [task.task_id for task in tasks]
        if configured_ids is not None and actual_ids != list(configured_ids):
            raise ValueError(
                "Runner task list must match experiment.tasks exactly; "
                f"configured={list(configured_ids)!r}, actual={actual_ids!r}"
            )
        languages = list(experiment.get("languages", []))
        agents = list(experiment.get("agents", []))
        models = list(experiment.get("models", []))
        if not languages or not agents or not models:
            raise ValueError("Experiment matrix dimensions must be non-empty")
        if len(set(languages)) != len(languages):
            raise ValueError("Experiment languages must not contain duplicates")
        for language in languages:
            if language not in {"english", "hindi", "hinglish"}:
                raise ValueError(f"Unsupported language: {language}")
        for agent in agents:
            if agent not in self.agents:
                raise ValueError(f"Configured agent is missing: {agent}")
        for model in models:
            if model not in self.models:
                raise ValueError(f"Configured model is missing: {model}")
        revision = self._dataset_revision()
        base_seed = int(experiment.get("seed", 0))
        cells: list[Cell] = []
        for task in tasks:
            for language in languages:
                for agent in agents:
                    for model in models:
                        for repetition in range(int(experiment["repetitions"])):
                            model_id = str(self.models[model].get("model", model))
                            seed = base_seed + repetition
                            cell_id = stable_cell_id(
                                self.config.experiment_id,
                                revision,
                                task.task_id,
                                language,
                                agent,
                                model_id,
                                repetition,
                            )
                            cells.append(
                                Cell(task, language, agent, model, repetition, seed, cell_id)
                            )
        return cells

    def validate_matrix(self, tasks: list[TaskDefinition]) -> dict[str, int]:
        cells = self.plan_cells(tasks)
        expected = len(cells)
        if len({cell.cell_id for cell in cells}) != expected:
            raise ValueError("Experiment matrix generated duplicate stable cell identities")
        return {
            "tasks": len(tasks),
            "cells": expected,
            "languages": len(self.config.experiment["languages"]),
            "agents": len(self.config.experiment["agents"]),
            "models": len(self.config.experiment["models"]),
            "repetitions": int(self.config.experiment["repetitions"]),
        }

    def run(
        self,
        tasks: list[TaskDefinition],
        *,
        resume: bool = True,
        max_cells: int | None = None,
    ) -> list[dict[str, Any]]:
        cells = self.plan_cells(tasks)
        experiment = self.config.experiment
        self.store.ensure_experiment(
            self.config.experiment_id,
            self._safe_config(),
            self._experiment_manifest(),
            utc_now(),
        )
        self.store.recover_interrupted(self.config.experiment_id, utc_now())
        for task in tasks:
            self.store.add_task(task.task_id, task.category, task.source_task)
            for language in ("english", "hindi", "hinglish"):
                self.store.add_language_variant(
                    task.task_id,
                    language,
                    task.instruction_for(language),
                    "canonical" if language == "english" else "unreviewed",
                )
        for cell in cells:
            model_id = str(self.models[cell.model].get("model", cell.model))
            self.store.ensure_cell(
                cell_id=cell.cell_id,
                experiment_id=self.config.experiment_id,
                task_id=cell.task.task_id,
                language=cell.language,
                model=model_id,
                agent=cell.agent,
                repetition=cell.repetition,
                seed=cell.seed,
            )
        self.store.commit()

        random.Random(int(experiment.get("seed", 0))).shuffle(cells)
        results: list[dict[str, Any]] = []
        processed = 0
        for cell in cells:
            row = self.store.get_cell(cell.cell_id)
            if resume and row is not None and row["status"] == "completed":
                results.append(self._result_from_row(row))
                continue
            if max_cells is not None and processed >= max_cells:
                continue
            results.append(self._run_cell(cell))
            processed += 1
        return results

    def completion_summary(self, tasks: list[TaskDefinition]) -> dict[str, int]:
        expected = len(self.plan_cells(tasks))
        rows = self.store.rows(self.config.experiment_id)
        completed = sum(1 for row in rows if row["status"] == "completed")
        gradable = sum(
            1
            for row in rows
            if row["status"] == "completed"
            and self.store.connection.execute(
                "SELECT 1 FROM grade WHERE run_id = ? LIMIT 1", (row["run_id"],)
            ).fetchone()
        )
        infrastructure = sum(1 for row in rows if row["status"] == "infrastructure_error")
        pending = sum(1 for row in rows if row["status"] in {"pending", "running"})
        return {
            "expected": expected,
            "completed": completed,
            "gradable": gradable,
            "infrastructure_error": infrastructure,
            "pending": pending,
        }

    def _run_cell(self, cell: Cell) -> dict[str, Any]:
        model_config = self.models[cell.model]
        max_retries = int(self.config.inference.get("max_retries", 2))
        row = self.store.get_cell(cell.cell_id)
        previous_attempts = int(row["attempt_count"] or 0) if row else 0
        trace: list[dict[str, Any]] = []
        overall_started = monotonic()
        final_response: AgentResponse | None = None
        final_grade: GradeResult | None = None
        final_workspace_hash: str | None = None
        final_agent_time: float | None = None
        final_error: Exception | None = None
        final_status = "infrastructure_error"
        final_attempt_id: str | None = None
        attempts_made = 0

        for retry_index in range(max_retries + 1):
            attempt_no = previous_attempts + retry_index + 1
            attempts_made += 1
            attempt_id = f"{cell.cell_id}-attempt-{attempt_no}"
            final_attempt_id = attempt_id
            attempt_started = utc_now()
            attempt_elapsed_started = monotonic()
            self.store.start_cell(cell.cell_id, attempt_started, attempt_no)
            self.store.start_attempt(attempt_id, cell.cell_id, attempt_no, attempt_started)
            trace.append(
                self._event(
                    cell.cell_id,
                    0,
                    "attempt_start",
                    {"attempt_id": attempt_id, "attempt_no": attempt_no},
                )
            )
            try:
                response, grade, attempt_trace, workspace_hash, agent_time = self._execute_attempt(
                    cell, model_config, attempt_id
                )
                trace.extend(attempt_trace)
                final_response = response
                final_grade = grade
                final_workspace_hash = workspace_hash
                final_agent_time = agent_time
                final_status = "completed"
                self.store.finish_attempt(
                    attempt_id,
                    status="completed",
                    end_time=utc_now(),
                    agent_time=agent_time,
                    end_to_end_time=monotonic() - attempt_elapsed_started,
                )
                break
            except Exception as exc:
                final_error = exc
                error_text = redact(str(exc), self._redaction_secrets(model_config))
                error_event = self._event(
                    cell.cell_id,
                    0,
                    "runner_error",
                    {"attempt_id": attempt_id, "type": type(exc).__name__, "message": error_text},
                )
                trace.append(error_event)
                transient = is_transient_endpoint_error(exc)
                self.store.finish_attempt(
                    attempt_id,
                    status="transient_error" if transient else "infrastructure_error",
                    end_time=utc_now(),
                    error_type=type(exc).__name__,
                    error_message=error_text,
                    end_to_end_time=monotonic() - attempt_elapsed_started,
                )
                if not transient or retry_index >= max_retries:
                    break
            finally:
                self._append_trace(cell.cell_id, trace)
                self.store.commit()

        elapsed = monotonic() - overall_started
        success = bool(final_grade and final_grade.success)
        if final_status == "completed":
            self.store.add_grade(
                cell.cell_id,
                "task_grader",
                success,
                json.dumps(
                    {
                        "returncode": final_grade.returncode if final_grade else None,
                        "stdout": (final_grade.stdout if final_grade else "")[-12000:],
                        "stderr": (final_grade.stderr if final_grade else "")[-12000:],
                    },
                    ensure_ascii=False,
                ),
            )
        usage = final_response.usage if final_response else {}
        metadata = final_response.metadata if final_response else {}
        self.store.finish_cell(
            cell.cell_id,
            end_time=utc_now(),
            status=final_status,
            success=success if final_status == "completed" else False,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            initial_prompt_tokens=metadata.get("initial_prompt_tokens"),
            tool_calls=metadata.get("tool_calls", 0),
            failed_tool_calls=metadata.get("failed_tool_calls", 0),
            execution_time=final_agent_time,
            agent_time=final_agent_time,
            end_to_end_time=elapsed,
            workspace_sha256=final_workspace_hash,
            trace_path=str(
                self.config.root / self.config.storage["traces_dir"] / f"{cell.cell_id}.jsonl"
            ),
            model_metadata=public_model_config(model_config),
            metadata={
                "attempt_id": final_attempt_id,
                "attempts": previous_attempts + attempts_made,
                "trace_event_count": len(trace),
                "error": redact(str(final_error), self._redaction_secrets(model_config))
                if final_error
                else None,
            },
        )
        self._append_trace(cell.cell_id, trace)
        result = {
            "run_id": cell.cell_id,
            "cell_id": cell.cell_id,
            "task_id": cell.task.task_id,
            "language": cell.language,
            "agent": cell.agent,
            "model": cell.model,
            "repetition": cell.repetition,
            "status": final_status,
            "success": success if final_status == "completed" else None,
            "execution_time": final_agent_time,
            "end_to_end_time": elapsed,
        }
        result_path = self.config.root / self.config.storage["results_dir"] / f"{cell.cell_id}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.store.commit()
        return result

    def _execute_attempt(
        self,
        cell: Cell,
        model_config: dict[str, Any],
        attempt_id: str,
    ) -> tuple[AgentResponse, GradeResult, list[dict[str, Any]], str, float]:
        task = cell.task
        task_root = self.config.task_root / task.task_id
        source_workspace: Path | None = None
        fixtures: Path | None = None
        if task.upstream is not None:
            fixtures = task_root / task.upstream.source_dir / task.upstream.fixtures_dir
        else:
            source_workspace = task_root / task.environment.workspace
        trace: list[dict[str, Any]] = [
            self._event(
                cell.cell_id,
                0,
                "run_start",
                {
                    "task_id": task.task_id,
                    "language": cell.language,
                    "agent": cell.agent,
                    "model": cell.model,
                    "repetition": cell.repetition,
                    "attempt_id": attempt_id,
                },
            )
        ]
        secrets = self._redaction_secrets(model_config)
        with WorkspaceSandbox(
            source_workspace=source_workspace,
            image=self.config.sandbox["image"],
            mode=self.config.sandbox["mode"],
            fixtures=fixtures,
            network=self.config.sandbox.get("network", "none"),
        ) as sandbox:
            agent = build_agent(
                cell.agent,
                self.agents[cell.agent],
                model_config,
                int(self.config.inference.get("timeout_seconds", task.limits.timeout_seconds)),
            )
            workspace_for_prompt = "/workspace" if self.config.sandbox["mode"] == "docker" else sandbox.workspace
            request = AgentRequest(
                instruction=render_instruction(task.instruction_for(cell.language), workspace_for_prompt),
                workspace=str(sandbox.workspace),
                system_prompt=SYSTEM_PROMPT,
                model=model_config["model"],
                temperature=float(self.config.generation["temperature"]),
                top_p=float(self.config.generation["top_p"]),
                max_tokens=int(self.config.generation["max_tokens"]),
                max_steps=task.limits.max_steps,
                command_runner=sandbox.run_command if self.config.sandbox["mode"] == "docker" else None,
                command_timeout_seconds=task.limits.timeout_seconds,
            )
            agent_started = monotonic()
            try:
                response = agent.run(request)
            finally:
                agent.close()
            agent_time = monotonic() - agent_started
            for event in response.metadata.get("events", []):
                event_type = event.get("event_type", "agent_event")
                event_payload = redact(
                    {key: value for key, value in event.items() if key != "event_type"}, secrets
                )
                trace_event = self._event(cell.cell_id, int(event.get("step", 0)), event_type, event_payload)
                trace.append(trace_event)
                self.store.add_event(
                    run_id=cell.cell_id,
                    attempt_id=attempt_id,
                    step=trace_event["step"],
                    event_type=event_type,
                    timestamp=trace_event["timestamp"],
                    tool=event.get("tool"),
                    arguments=redact(event.get("arguments"), secrets),
                    result=redact(event.get("result"), secrets),
                )
            trace.append(
                self._event(
                    cell.cell_id,
                    0,
                    "agent_result",
                    {
                        "attempt_id": attempt_id,
                        "completed": response.completed,
                        "usage": response.usage,
                        "metadata": {
                            key: value
                            for key, value in response.metadata.items()
                            if key != "events"
                        },
                    },
                )
            )
            final_workspace_hash = workspace_sha256(sandbox.workspace)
            if task.upstream is not None:
                grade = run_upstream_oracle(
                    task_root / task.upstream.source_dir,
                    task.upstream.oracle_module,
                    sandbox.workspace,
                    task.upstream.expected_outcome_score,
                    oracle_runner=sandbox.run_oracle if self.config.sandbox["mode"] == "docker" else None,
                    timeout_seconds=task.limits.timeout_seconds,
                )
            else:
                assert task.judge is not None
                command_runner = (
                    sandbox.run_grader_command if self.config.sandbox["mode"] == "docker" else None
                )
                grade = run_grader(
                    sandbox.workspace,
                    task.judge.command,
                    task.judge.workdir,
                    task.judge.expected_exit_code,
                    task.limits.timeout_seconds,
                    command_runner=command_runner,
                )
            return response, grade, trace, final_workspace_hash, agent_time

    def _dataset_revision(self) -> str:
        manifest_path = self.config.root / self.config.experiment.get(
            "dataset_manifest", "benchmark/task_selection.yaml"
        )
        try:
            import yaml

            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            return str(data.get("source", {}).get("revision", "unknown"))
        except (OSError, ValueError):
            return "unknown"

    def _experiment_manifest(self) -> dict[str, Any]:
        path = self.config.root / self.config.experiment.get(
            "dataset_manifest", "benchmark/task_selection.yaml"
        )
        try:
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return {"path": str(path)}

    def _safe_config(self) -> dict[str, Any]:
        data = json.loads(json.dumps(self.config.data, ensure_ascii=False))
        if isinstance(data.get("inference"), dict):
            data["inference"].pop("api_key", None)
        return data

    @staticmethod
    def _redaction_secrets(model_config: dict[str, Any]) -> tuple[str, ...]:
        values = [model_config.get("api_key"), model_config.get("base_url")]
        return tuple(value for value in values if isinstance(value, str) and value)

    def _append_trace(self, cell_id: str, trace: list[dict[str, Any]]) -> None:
        path = self.config.root / self.config.storage["traces_dir"] / f"{cell_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in trace:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _event(run_id: str, step: int, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "step": step,
            "event_type": event_type,
            "timestamp": utc_now(),
            "data": data,
        }

    @staticmethod
    def _result_from_row(row: Any) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "cell_id": row["cell_id"] or row["run_id"],
            "task_id": row["task_id"],
            "language": row["language"],
            "agent": row["agent"],
            "model": row["model"],
            "repetition": row["repetition"],
            "status": row["status"],
            "success": bool(row["success"]),
            "execution_time": row["execution_time"],
            "end_to_end_time": row["end_to_end_time"],
        }
