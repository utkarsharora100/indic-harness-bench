from __future__ import annotations

import hashlib
import json
import random
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any

from agents.base import AgentRequest, AgentResponse
from agents.factory import build_agent
from benchmark.models import Language, TaskDefinition
from benchmark.upstream import (
    after_round_runtime,
    cleanup_runtime,
    prepare_runtime,
    render_runtime_template,
)
from runner.config import ExperimentConfig
from runner.grader import GradeResult, GraderInfrastructureError, run_grader, run_upstream_oracle
from runner.inference import InferenceTransientError, public_model_config
from runner.log import RunStore
from runner.proxy import InferenceProxy
from runner.proxy_sidecar import ProxySidecar
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
        self.proxies: dict[str, InferenceProxy] = {}
        proxy_config = config.inference.get("proxy") or {}
        if proxy_config.get("enabled"):
            for model_name, model_config in list(self.models.items()):
                if model_config.get("provider") != "university_gpu":
                    continue
                if not model_config.get("base_url") or not model_config.get("api_key"):
                    raise ValueError(f"Model {model_name} cannot use the proxy without private endpoint settings")
                proxy = InferenceProxy(
                    str(model_config["base_url"]),
                    str(model_config["api_key"]),
                    str(model_config["model"]),
                    public_model=str(proxy_config.get("public_model", "phase1-university-model")),
                    max_calls_per_cell=int(proxy_config.get("max_calls_per_cell", 40)),
                    temperature=float(config.generation.get("temperature", 0.0)),
                    top_p=float(config.generation.get("top_p", 1.0)),
                    max_tokens=int(config.generation.get("max_tokens", 2048)),
                ).start()
                self.proxies[model_name] = proxy
                public = dict(model_config)
                public["base_url"] = proxy.base_url
                public["container_base_url"] = proxy.container_base_url
                public["api_key"] = proxy.client_key
                public["model"] = str(proxy_config.get("public_model", "phase1-university-model"))
                public["proxy"] = True
                self.models[model_name] = public
        database = config.root / config.storage["database"]
        self.store = RunStore(database)
        (config.root / config.storage["traces_dir"]).mkdir(parents=True, exist_ok=True)
        (config.root / config.storage["results_dir"]).mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.store.commit()
        self.store.close()
        for proxy in self.proxies.values():
            proxy.close()

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

        cells = self._execution_order(cells, int(experiment.get("seed", 0)))
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

    @staticmethod
    def _execution_order(cells: list[Cell], seed: int) -> list[Cell]:
        """Shuffle condition order within task/repetition blocks.

        This controls endpoint drift without treating temperature-zero repeats
        as independent stochastic observations. The resulting order is fully
        determined by the experiment seed and stable cell identities.
        """
        by_block: dict[tuple[str, int], list[Cell]] = {}
        for cell in cells:
            by_block.setdefault((cell.task.task_id, cell.repetition), []).append(cell)
        ordered: list[Cell] = []
        for block_index, key in enumerate(sorted(by_block)):
            block = list(by_block[key])
            random.Random(seed + block_index).shuffle(block)
            ordered.extend(block)
        return ordered

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
        final_workspace_archive: str | None = None
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
                attempt_trace: list[dict[str, Any]] = []
                response, grade, attempt_trace, workspace_hash, agent_time, archive_path = self._execute_attempt(
                    cell, model_config, attempt_id, attempt_trace
                )
                trace.extend(attempt_trace)
                final_response = response
                final_grade = grade
                final_workspace_hash = workspace_hash
                final_workspace_archive = archive_path
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
                trace.extend(attempt_trace)
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
                self._append_trace(cell.cell_id, trace, model_config)
                self.store.commit()

        elapsed = monotonic() - overall_started
        success = bool(final_grade and final_grade.success)
        if final_status == "completed":
            grade_payload = {
                "returncode": final_grade.returncode if final_grade else None,
                "success": bool(final_grade and final_grade.success),
                "outcome_score": final_grade.outcome_score if final_grade else None,
                "stdout": final_grade.stdout if final_grade else "",
                "stderr": final_grade.stderr if final_grade else "",
                "details": final_grade.details if final_grade else None,
                "infrastructure_error": bool(final_grade and final_grade.infrastructure_error),
                "error_type": final_grade.error_type if final_grade else None,
            }
            self.store.add_grade(
                cell.cell_id,
                "task_grader",
                success,
                json.dumps(redact(grade_payload, self._redaction_secrets(model_config)), ensure_ascii=False),
                score=final_grade.outcome_score if final_grade else None,
                status="completed" if final_grade and not final_grade.infrastructure_error else "invalid",
            )
        usage = final_response.usage if final_response else {}
        metadata = final_response.metadata if final_response else {}
        self.store.finish_cell(
            cell.cell_id,
            end_time=utc_now(),
            status=final_status,
            success=success if final_status == "completed" else None,
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
                "outcome_score": final_grade.outcome_score if final_grade else None,
                "workspace_archive": final_workspace_archive,
                "agent_completed": final_response.completed if final_response else None,
                "agent_stop_reason": metadata.get("stop_reason"),
                "last_model_finish_reason": metadata.get("last_model_finish_reason"),
                "trace_complete": final_status == "completed",
                "grade_status": (
                    "completed"
                    if final_grade and not final_grade.infrastructure_error
                    else "missing"
                ),
            },
        )
        self._append_trace(cell.cell_id, trace, model_config)
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
            "workspace_archive": final_workspace_archive,
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
        trace_sink: list[dict[str, Any]] | None = None,
    ) -> tuple[AgentResponse, GradeResult, list[dict[str, Any]], str, float, str]:
        task = cell.task
        task_root = self.config.task_root / task.task_id
        source_workspace: Path | None = None
        fixtures: Path | None = None
        if task.upstream is not None:
            fixtures = task_root / task.upstream.source_dir / task.upstream.fixtures_dir
        else:
            source_workspace = task_root / task.environment.workspace
        trace: list[dict[str, Any]] = trace_sink if trace_sink is not None else []
        trace.append(
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
        )
        secrets = self._redaction_secrets(model_config)
        with WorkspaceSandbox(
            source_workspace=source_workspace,
            image=self.config.sandbox["image"],
            mode=self.config.sandbox["mode"],
            fixtures=fixtures,
            network=self.config.sandbox.get("network", "none"),
        ) as sandbox:
            runtime_state: dict[str, Any] = {}
            runtime_env: dict[str, str] = {}
            if task.upstream is not None:
                source_task_dir = task_root / task.upstream.source_dir
                runtime_state = prepare_runtime(
                    source_task_dir,
                    sandbox.root or sandbox.workspace.parent,
                    sandbox.workspace,
                    task.upstream.hooks_module,
                )
                runtime_env = {}
                for key, value in runtime_state.items():
                    if not isinstance(value, (str, int, float)):
                        continue
                    rendered_value = str(value)
                    if self.config.sandbox["mode"] == "docker":
                        path_value = Path(rendered_value)
                        if path_value.is_absolute() and path_value.is_relative_to(sandbox.workspace):
                            rendered_value = "/workspace/" + path_value.relative_to(sandbox.workspace).as_posix()
                    runtime_env[key] = rendered_value
                trace.append(
                    self._event(
                        cell.cell_id,
                        0,
                        "runtime_prepared",
                        {"keys": sorted(runtime_env), "hook": task.upstream.hooks_module},
                    )
                )
            sidecar: ProxySidecar | None = None
            agent_model_config = model_config
            if cell.agent != "react":
                host_proxy = self.proxies.get(cell.model)
                if host_proxy is None:
                    raise RuntimeError("Native harness requires the university model proxy")
                proxy_config = self.config.inference.get("proxy") or {}
                sidecar = ProxySidecar(
                    image=str(proxy_config.get("image", "indic-harness-proxy:phase1-pinned")),
                    internal_network=str(self.agents[cell.agent].get("network", "phase1-agent-net")),
                    egress_network=str(proxy_config.get("egress_network", "bridge")),
                    upstream_base_url=host_proxy.upstream_base_url,
                    upstream_key=host_proxy.upstream_key,
                    resolved_model=host_proxy.resolved_model,
                    public_model=host_proxy.public_model,
                    trace_root=self.config.root / self.config.storage["traces_dir"] / "sidecars",
                    cell_id=cell.cell_id,
                    max_calls=int(proxy_config.get("max_calls_per_cell", 40)),
                    temperature=float(self.config.generation.get("temperature", 0.0)),
                    top_p=float(self.config.generation.get("top_p", 1.0)),
                    max_tokens=int(self.config.generation.get("max_tokens", 2048)),
                ).start()
                agent_model_config = sidecar.model_config(model_config)
            try:
                agent = build_agent(
                    cell.agent,
                    self.agents[cell.agent],
                    agent_model_config,
                    int(task.limits.timeout_seconds),
                )
            except Exception:
                # A startup/configuration failure can happen after the
                # sidecar is already running. Do not leave its private-key
                # environment or stable DNS name behind for the next cell.
                if sidecar is not None:
                    sidecar.close()
                raise
            workspace_for_prompt = "/workspace" if self.config.sandbox["mode"] == "docker" else sandbox.workspace
            request = AgentRequest(
                instruction=render_runtime_template(
                    task.instruction_for(cell.language),
                    workspace=workspace_for_prompt,
                    runtime_env=runtime_env,
                ),
                workspace=str(sandbox.workspace),
                system_prompt=SYSTEM_PROMPT,
                # ReAct talks to the host proxy and uses the resolved model
                # internally. Native harnesses must receive only the proxy's
                # public alias; the sidecar owns the private upstream ID.
                model=agent_model_config.get("model", model_config["model"]),
                temperature=float(self.config.generation["temperature"]),
                top_p=float(self.config.generation["top_p"]),
                max_tokens=int(self.config.generation["max_tokens"]),
                max_steps=task.limits.max_steps,
                command_runner=sandbox.run_command if self.config.sandbox["mode"] == "docker" else None,
                command_timeout_seconds=task.limits.timeout_seconds,
                metadata={"cell_id": cell.cell_id, "attempt_id": attempt_id, "session_id": attempt_id},
            )
            agent_started = monotonic()
            proxy = self.proxies.get(cell.model) if cell.agent == "react" else None
            sidecar_snapshot: dict[str, Any] | None = None
            if proxy is not None:
                proxy.begin_cell(cell.cell_id)
            try:
                try:
                    response = agent.run(request)
                finally:
                    agent.close()
                agent_time = monotonic() - agent_started
                if sidecar is not None:
                    try:
                        sidecar_snapshot = sidecar.seal()
                    except Exception as exc:
                        try:
                            sidecar_snapshot = sidecar.snapshot()
                        except Exception:
                            sidecar_snapshot = None
                        trace.append(self._event(cell.cell_id, 0, "agent_result", {
                            "attempt_id": attempt_id,
                            "completed": response.completed,
                            "usage": response.usage,
                            "metadata": {key: value for key, value in response.metadata.items()
                                         if key != "events"},
                        }))
                        trace.append(self._event(cell.cell_id, 0, "proxy_trace_error", {
                            "attempt_id": attempt_id,
                            "agent_completed": response.completed,
                            "agent_stop_reason": response.metadata.get("stop_reason"),
                            "error_type": type(exc).__name__,
                        }))
                        raise
                    final_reasons = [
                        choice.get("finish_reason")
                        for event in sidecar_snapshot.get("events", [])
                        if event.get("event_type") == "proxy_response"
                        for choice in (event.get("data", {}).get("choices") or [])
                        if isinstance(choice, dict) and choice.get("finish_reason")
                    ]
                    if final_reasons and final_reasons[-1] == "length":
                        response.completed = False
                        response.metadata["stop_reason"] = "max_tokens"
                    response.metadata["last_model_finish_reason"] = final_reasons[-1] if final_reasons else None
                active_proxy = proxy or sidecar
                if active_proxy is not None:
                    usage = (
                        proxy.cell_usage()
                        if proxy is not None
                        else (sidecar_snapshot or {}).get("usage", {})
                    )
                    if any(isinstance(value, int) and value > 0 for value in usage.values()):
                        response.usage = usage
                    response.metadata["proxy_calls"] = (
                        proxy._cell_calls if proxy is not None else (sidecar_snapshot or {}).get("calls", 0)
                    )
                if cell.agent != "react":
                    response.metadata["tool_calls"] = int(response.metadata.get("native_tool_calls", 0) or 0)
                    response.metadata["failed_tool_calls"] = int(
                        response.metadata.get("native_failed_tool_calls", 0) or 0
                    )
                for event in response.metadata.get("events", []):
                    event_type = event.get("event_type", "agent_event")
                    event_payload = redact(
                        {key: value for key, value in event.items() if key != "event_type"}, secrets
                    )
                    trace_event = self._event(
                        cell.cell_id, int(event.get("step", 0)), event_type, event_payload
                    )
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
                if task.upstream is not None:
                    runtime_state = after_round_runtime(
                        task_root / task.upstream.source_dir,
                        sandbox.root or sandbox.workspace.parent,
                        sandbox.workspace,
                        task.upstream.hooks_module,
                        runtime_state,
                        SimpleNamespace(
                            ok=response.completed,
                            completed=response.completed,
                            text=response.text,
                            metadata=response.metadata,
                        ),
                    )
                    trace.append(
                        self._event(
                            cell.cell_id,
                            0,
                            "runtime_after_round",
                            {"keys": sorted(runtime_state)},
                        )
                    )
                final_workspace_hash = workspace_sha256(sandbox.workspace)
                archive_dir = self.config.root / self.config.storage["results_dir"] / "workspaces"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"{attempt_id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    archive.add(sandbox.workspace, arcname="workspace", recursive=True)
                if task.upstream is not None:
                    grade = run_upstream_oracle(
                        task_root / task.upstream.source_dir,
                        task.upstream.oracle_module,
                        sandbox.workspace,
                        task.upstream.expected_outcome_score,
                        oracle_runner=sandbox.run_oracle
                        if self.config.sandbox["mode"] == "docker"
                        else None,
                        timeout_seconds=task.limits.timeout_seconds,
                    )
                else:
                    assert task.judge is not None
                    command_runner = (
                        sandbox.run_grader_command
                        if self.config.sandbox["mode"] == "docker"
                        else None
                    )
                    grade = run_grader(
                        sandbox.workspace,
                        task.judge.command,
                        task.judge.workdir,
                        task.judge.expected_exit_code,
                        task.limits.timeout_seconds,
                        command_runner=command_runner,
                    )
                return response, grade, trace, final_workspace_hash, agent_time, str(archive_path)
            finally:
                if proxy is not None:
                    for proxy_event in proxy.end_cell():
                        observable = redact(proxy_event.get("data", {}), secrets)
                        trace_event = self._event(
                            cell.cell_id,
                            0,
                            proxy_event["event_type"],
                            observable,
                        )
                        trace_event["timestamp"] = proxy_event.get(
                            "timestamp", trace_event["timestamp"]
                        )
                        trace.append(trace_event)
                        self.store.add_event(
                            run_id=cell.cell_id,
                            attempt_id=attempt_id,
                            step=0,
                            event_type=proxy_event["event_type"],
                            timestamp=trace_event["timestamp"],
                            result=observable,
                        )
                if sidecar is not None:
                    for sidecar_event in (sidecar_snapshot or {}).get("events", []):
                        observable = redact(sidecar_event.get("data", {}), secrets)
                        trace_event = self._event(
                            cell.cell_id,
                            0,
                            sidecar_event.get("event_type", "proxy_event"),
                            observable,
                        )
                        trace_event["timestamp"] = sidecar_event.get(
                            "timestamp", trace_event["timestamp"]
                        )
                        trace.append(trace_event)
                        self.store.add_event(
                            run_id=cell.cell_id,
                            attempt_id=attempt_id,
                            step=0,
                            event_type=trace_event["event_type"],
                            timestamp=trace_event["timestamp"],
                            result=observable,
                        )
                    sidecar.close()
                if task.upstream is not None:
                    cleanup_runtime(
                        task_root / task.upstream.source_dir,
                        sandbox.root or sandbox.workspace.parent,
                        sandbox.workspace,
                        task.upstream.hooks_module,
                        runtime_state,
                    )

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
        if self.config.experiment.get("version") == "corrected-v13":
            paths = {
                "dataset": self.config.experiment["dataset_manifest"],
                "translations": self.config.experiment["translations_manifest"],
                "agents": self.config.experiment["agents_manifest"],
                "model": self.config.inference["manifest"],
                "runtime": self.config.inference["runtime_manifest"],
                "calibration": self.config.inference["calibration_manifest"],
                "runner": "runner/runner.py",
                "react": "agents/react.py",
                "native": "agents/container.py",
                "proxy": "runner/proxy.py",
                "sidecar": "scripts/proxy_sidecar.py",
                "preflight": "runner/preflight.py",
                "calibration_code": "scripts/calibrate_pilot_budget.py",
                "preparation_code": "scripts/prepare_phase1.py",
                "grader": "runner/grader.py",
                "judge": "runner/judge.py",
                "upstream": "benchmark/upstream.py",
            }
            data["frozen_sha256"] = {
                name: hashlib.sha256((self.config.root / path).read_bytes()).hexdigest()
                for name, path in paths.items()
            }
        return data

    def assert_experiment_identity(self) -> None:
        """Reject scoring or resuming under changed v13 runtime provenance."""
        row = self.store.connection.execute(
            "SELECT config_json, manifest_json FROM experiment WHERE experiment_id = ?",
            (self.config.experiment_id,),
        ).fetchone()
        if row is None:
            raise ValueError("No completed experiment exists to judge")
        expected_config = json.dumps(self._safe_config(), ensure_ascii=False, sort_keys=True)
        expected_manifest = json.dumps(self._experiment_manifest(), ensure_ascii=False, sort_keys=True)
        if row["config_json"] != expected_config or row["manifest_json"] != expected_manifest:
            raise ValueError("Experiment dataset, model, image, calibration, or code identity changed")

    def _redaction_secrets(self, model_config: dict[str, Any]) -> tuple[str, ...]:
        values = [model_config.get("api_key"), model_config.get("base_url")]
        for proxy in self.proxies.values():
            values.extend((proxy.upstream_key, proxy.upstream_base_url, proxy.resolved_model))
        return tuple(value for value in values if isinstance(value, str) and value)

    def _append_trace(self, cell_id: str, trace: list[dict[str, Any]], model_config: dict[str, Any]) -> None:
        path = self.config.root / self.config.storage["traces_dir"] / f"{cell_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        secrets = self._redaction_secrets(model_config)
        with path.open("w", encoding="utf-8") as handle:
            for event in trace:
                handle.write(json.dumps(redact(event, secrets), ensure_ascii=False) + "\n")

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
