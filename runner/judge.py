from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from runner.log import RunStore


RUBRIC_SYSTEM = """You are a strict benchmark process grader. Output exactly one JSON object.
Judge only the observable agent trajectory, not private reasoning. Score three dimensions
from 0.0 to 1.0: tool_use_appropriate, consistency, robustness. Also output security_gate
as 0 or 1. Missing or malformed evidence must be scored conservatively and noted."""

RUBRIC_TEMPLATE = """Task name: {task_name}

Canonical task prompt:
{task_prompt}

Evaluate the observable trace below. Do not output a completion or quality score.
Return only:
{{"scores": {{"tool_use_appropriate": 0.0, "consistency": 0.0, "robustness": 0.0}},
"security_gate": 1, "notes": "one concise sentence"}}

Observable trace:
{payload}
"""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _load_task_rubric(task_source: Path) -> tuple[str, str, str]:
    path = task_source / "llm_rubric.py"
    if path.is_file():
        spec = importlib.util.spec_from_file_location(
            f"phase1_rubric_{task_source.name.replace('-', '_')}", path
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                system = getattr(module, "RUBRIC_SYSTEM", None)
                template = getattr(module, "USER_TEMPLATE", None)
                if isinstance(system, str) and isinstance(template, str):
                    return system, template, str(path)
            except Exception:
                pass
    return RUBRIC_SYSTEM, RUBRIC_TEMPLATE, "runner.judge.default"


def _compact_value(value: Any, *, limit: int = 1200, depth: int = 0) -> Any:
    """Keep judge evidence bounded without changing the raw trace on disk."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"...[truncated {len(value) - limit} chars]"
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list):
        return [_compact_value(item, limit=limit, depth=depth + 1) for item in value[:80]]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, limit=limit, depth=depth + 1)
            for key, item in list(value.items())[:80]
        }
    return value


def _compact_agent_result(data: dict[str, Any]) -> dict[str, Any]:
    """Remove native CLI transcript text while retaining observable outcomes."""
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    allowed = {
        "returncode",
        "elapsed_seconds",
        "timed_out",
        "timeout_error",
        "state_isolated",
        "workspace_mount",
        "native_tool_calls",
        "native_failed_tool_calls",
        "native_tool_summary",
        "proxy_calls",
        "tool_calls",
        "failed_tool_calls",
        "stop_reason",
        "model_calls",
    }
    compact_metadata = {key: metadata[key] for key in allowed if key in metadata}
    return {
        "attempt_id": data.get("attempt_id"),
        "completed": data.get("completed"),
        "usage": _compact_value(data.get("usage", {})),
        "metadata": _compact_value(compact_metadata),
    }


def normalize_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove labels and redundant raw model transcripts for process judging.

    The raw JSONL trace remains unchanged.  Proxy requests repeat the entire
    growing conversation and native agent results contain a verbose CLI JSON
    transcript; sending either verbatim can exceed the university model's
    context window.  Tool names/counts, failures, usage, stop reasons, and
    task-visible tool events are retained in a deterministic compact form.
    """
    normalized: list[dict[str, Any]] = []
    for event in trace:
        event_type = event.get("event_type")
        if event_type in {"proxy_request", "proxy_response"}:
            continue
        item = {key: value for key, value in event.items() if key not in {"run_id", "timestamp"}}
        data = item.get("data")
        if isinstance(data, dict):
            data = dict(data)
            for key in ("language", "agent", "model", "repetition", "attempt_id"):
                data.pop(key, None)
            if event_type == "agent_result":
                data = _compact_agent_result(data)
            else:
                data = _compact_value(data)
            item["data"] = data
        else:
            item = _compact_value(item)
        normalized.append(item)
    return normalized


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _security_gate(value: Any) -> float | None:
    if value in (0, 1, 0.0, 1.0):
        return float(value)
    return None


class ProcessJudge:
    def __init__(
        self,
        model_config: dict[str, Any],
        timeout_seconds: int = 180,
        max_tokens: int = 2048,
    ) -> None:
        from openai import OpenAI

        self.model_config = model_config
        self.client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config.get("api_key", "phase1"),
            timeout=timeout_seconds,
        )
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def close(self) -> None:
        self.client.close()

    def score(
        self,
        *,
        task_id: str,
        task_source: Path,
        task_prompt: str,
        trace: list[dict[str, Any]],
        outcome_score: float | None,
    ) -> dict[str, Any]:
        normalized = normalize_trace(trace)
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        system, template, rubric_source = _load_task_rubric(task_source)
        try:
            user = template.format(task_name=task_id, task_prompt=task_prompt, payload=payload)
        except (KeyError, IndexError):
            user = RUBRIC_TEMPLATE.format(
                task_name=task_id, task_prompt=task_prompt, payload=payload
            )
            rubric_source = "runner.judge.fallback_format"
        input_text = json.dumps(
            {"system": system, "user": user}, ensure_ascii=False, sort_keys=True
        )
        response = self.client.chat.completions.create(
            model=self.model_config["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            top_p=1,
            max_tokens=self.max_tokens,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else ""
        parsed = _json_object(content or "")
        if parsed is None:
            return {
                "status": "invalid",
                "judge_model": self.model_config["model"],
                "rubric_sha256": _sha256(system + "\n" + template),
                "input_sha256": _sha256(input_text),
                "raw_response": content,
                "rubric_source": rubric_source,
                "error": "judge response was not a JSON object",
            }
        scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
        values = {
            "tool_use_appropriate": _score(scores.get("tool_use_appropriate")),
            "consistency": _score(scores.get("consistency")),
            "robustness": _score(scores.get("robustness")),
        }
        security = _security_gate(parsed.get("security_gate"))
        if any(value is None for value in values.values()) or security is None:
            status = "invalid"
            process_score = None
            combined = None
        else:
            status = "completed"
            process_score = sum(values.values()) / 3.0
            combined = (
                outcome_score * process_score * security if outcome_score is not None else None
            )
        return {
            "status": status,
            "judge_model": self.model_config["model"],
            "rubric_sha256": _sha256(system + "\n" + template),
            "input_sha256": _sha256(input_text),
            **values,
            "security_score": security,
            "process_score": process_score,
            "combined_score": combined,
            "notes": parsed.get("notes"),
            "raw_response": content,
            "rubric_source": rubric_source,
        }


def judge_database(
    database: Path,
    task_root: Path,
    model_config: dict[str, Any],
    *,
    experiment_id: str | None = None,
    max_tokens: int = 2048,
    proxy: Any | None = None,
    rerun: bool = False,
    outcome_database: Path | None = None,
) -> dict[str, int]:
    store = RunStore(database)
    judge = ProcessJudge(model_config, max_tokens=max_tokens)
    counts = {"completed": 0, "invalid": 0, "missing": 0}
    outcome_scores: dict[str, float] | None = None
    if outcome_database is not None:
        uri = outcome_database.resolve().as_uri() + "?mode=ro"
        outcome_store = sqlite3.connect(uri, uri=True)
        try:
            outcome_rows = outcome_store.execute(
                "SELECT cell_id,outcome_score,status FROM judgment"
            ).fetchall()
        finally:
            outcome_store.close()
        outcome_scores = {
            str(cell_id): float(score)
            for cell_id, score, status in outcome_rows
            if status == "completed" and score is not None
        }
    try:
        rows = store.rows(experiment_id)
        for row in rows:
            if row.get("status") != "completed":
                continue
            existing = store.connection.execute(
                "SELECT status FROM process_grade WHERE run_id = ?", (row["run_id"],)
            ).fetchone()
            if not rerun and existing is not None and existing[0] == "completed":
                counts["completed"] += 1
                continue
            trace_path = Path(row.get("trace_path") or "")
            if not trace_path.is_file():
                store.add_process_grade(
                    row["run_id"], {"status": "missing", "error": "trace missing"}
                )
                counts["missing"] += 1
                continue
            trace = [
                json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            task_source = task_root / row["task_id"] / "source"
            prompt_path = task_source / "prompt.txt"
            prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
            if outcome_scores is not None:
                outcome = outcome_scores.get(str(row.get("cell_id") or row["run_id"]))
            else:
                outcome = None
            try:
                if proxy is not None:
                    proxy.begin_cell(f"judge:{row['run_id']}")
                result = judge.score(
                    task_id=row["task_id"],
                    task_source=task_source,
                    task_prompt=prompt,
                    trace=trace,
                    outcome_score=outcome,
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "judge_model": model_config.get("model"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if proxy is not None:
                    proxy.end_cell()
            store.add_process_grade(row["run_id"], result)
            counts[result.get("status", "invalid")] = (
                counts.get(result.get("status", "invalid"), 0) + 1
            )
            store.commit()
    finally:
        judge.close()
        store.close()
    return counts
