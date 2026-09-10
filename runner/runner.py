from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from agents.base import AgentRequest
from agents.factory import build_agent
from benchmark.models import Language, TaskDefinition
from runner.config import ExperimentConfig
from runner.grader import run_grader
from runner.log import RunStore
from runner.sandbox import WorkspaceSandbox

SYSTEM_PROMPT = (
    "You are operating in a controlled benchmark workspace. "
    "Complete the user's task by using the available tools and commands. "
    "Do not access files outside the workspace. Stop when the required workspace state is complete."
)


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

    def run(self, tasks: list[TaskDefinition]) -> list[dict]:
        experiment = self.config.experiment
        seed = int(experiment.get("seed", 0))
        cells = [
            (task, language, agent, model, repetition)
            for task in tasks
            for language in experiment["languages"]
            for agent in experiment["agents"]
            for model in experiment["models"]
            for repetition in range(int(experiment["repetitions"]))
        ]
        random.Random(seed).shuffle(cells)
        return [self._run_cell(task, language, agent, model, repetition, seed) for
                task, language, agent, model, repetition in cells]

    def _run_cell(
        self,
        task: TaskDefinition,
        language: Language,
        agent_name: str,
        model_name: str,
        repetition: int,
        seed: int,
    ) -> dict:
        run_id = uuid.uuid4().hex
        start_time = datetime.now(timezone.utc)
        started = monotonic()
        trace: list[dict] = []
        response = None
        grade = None

        self.store.add_task(task.task_id, task.category, task.source_task)
        for current_language in ("english", "hindi", "hinglish"):
            self.store.add_language_variant(
                task.task_id,
                current_language,
                task.instruction_for(current_language),
            )

        model_config = self.models[model_name]
        self.store.start_run(
            run_id, task.task_id, language, model_config["model"], agent_name,
            seed + repetition, start_time.isoformat(),
        )

        try:
            source_workspace = (
                self.config.root / "benchmark" / "tasks" / task.task_id / task.environment.workspace
            )
            with WorkspaceSandbox(
                source_workspace=source_workspace,
                image=self.config.sandbox["image"],
                mode=self.config.sandbox["mode"],
            ) as sandbox:
                agent = build_agent(
                    agent_name,
                    self.agents[agent_name],
                    model_config,
                    task.limits.timeout_seconds,
                    command_runner=sandbox.run_command if self.config.sandbox["mode"] == "docker" else None,
                )
                request = AgentRequest(
                    instruction=task.instruction_for(language),
                    workspace=str(sandbox.workspace),
                    system_prompt=SYSTEM_PROMPT,
                    model=model_config["model"],
                    temperature=float(self.config.generation["temperature"]),
                    top_p=float(self.config.generation["top_p"]),
                    max_tokens=int(self.config.generation["max_tokens"]),
                    max_steps=task.limits.max_steps,
                    command_runner=sandbox.run_command if self.config.sandbox["mode"] == "docker" else None,
                )
                trace.append(self._event(run_id, 0, "run_start", {
                    "task_id": task.task_id,
                    "language": language,
                    "agent": agent_name,
                    "model": model_name,
                    "repetition": repetition,
                }))
                response = agent.run(request)
                agent.close()
                for event in response.metadata.get("events", []):
                    trace_event = self._event(
                        run_id, event.get("step", 0), event["event_type"], event
                    )
                    trace.append(trace_event)
                    self.store.add_event(
                        run_id=run_id,
                        step=trace_event["step"],
                        event_type=trace_event["event_type"],
                        timestamp=trace_event["timestamp"],
                        tool=event.get("tool"),
                        arguments=event.get("arguments"),
                        result=event.get("result"),
                    )
                trace.append(self._event(run_id, 0, "agent_result", {
                    "completed": response.completed,
                    "text": response.text,
                    "usage": response.usage,
                    "metadata": response.metadata,
                }))

                grade = run_grader(
                    sandbox.workspace,
                    task.judge.command,
                    task.judge.workdir,
                    task.judge.expected_exit_code,
                    task.limits.timeout_seconds,
                    command_runner=sandbox.run_command if self.config.sandbox["mode"] == "docker" else None,
                )
                self.store.add_grade(
                    run_id,
                    "task_grader",
                    grade.success,
                    json.dumps({
                        "returncode": grade.returncode,
                        "stdout": grade.stdout[-12000:],
                        "stderr": grade.stderr[-12000:],
                    }, ensure_ascii=False),
                )
        except Exception as exc:
            trace.append(self._event(run_id, 0, "runner_error", {
                "type": type(exc).__name__, "message": str(exc)
            }))
            raise
        finally:
            elapsed = monotonic() - started
            usage = response.usage if response else {}
            success = bool(grade and grade.success)
            self.store.finish_run(
                run_id,
                end_time=datetime.now(timezone.utc).isoformat(),
                success=success,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                tool_calls=int((response.metadata if response else {}).get("tool_calls", 0)),
                failed_tool_calls=int((response.metadata if response else {}).get("failed_tool_calls", 0)),
                execution_time=elapsed,
                metadata={"trace_event_count": len(trace), "repetition": repetition},
            )
            trace_path = self.config.root / self.config.storage["traces_dir"] / f"{run_id}.jsonl"
            with trace_path.open("w", encoding="utf-8") as handle:
                for event in trace:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            result_path = self.config.root / self.config.storage["results_dir"] / f"{run_id}.json"
            result_path.write_text(json.dumps({
                "run_id": run_id,
                "task_id": task.task_id,
                "language": language,
                "agent": agent_name,
                "model": model_name,
                "repetition": repetition,
                "success": success,
                "execution_time": elapsed,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.commit()

        return {
            "run_id": run_id,
            "task_id": task.task_id,
            "language": language,
            "agent": agent_name,
            "model": model_name,
            "success": success,
            "execution_time": elapsed,
        }

    @staticmethod
    def _event(run_id: str, step: int, event_type: str, data: dict) -> dict:
        return {
            "run_id": run_id,
            "step": step,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
