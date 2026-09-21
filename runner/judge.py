from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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


def normalize_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove condition identifiers while retaining externally observable evidence."""
    normalized: list[dict[str, Any]] = []
    for event in trace:
        item = {key: value for key, value in event.items() if key not in {"run_id", "timestamp"}}
        data = item.get("data")
        if isinstance(data, dict):
            data = dict(data)
            for key in ("language", "agent", "model", "repetition", "attempt_id"):
                data.pop(key, None)
            item["data"] = data
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
            user = RUBRIC_TEMPLATE.format(task_name=task_id, task_prompt=task_prompt, payload=payload)
            rubric_source = "runner.judge.fallback_format"
        input_text = json.dumps({"system": system, "user": user}, ensure_ascii=False, sort_keys=True)
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
                outcome_score * process_score * security
                if outcome_score is not None
                else None
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
) -> dict[str, int]:
    store = RunStore(database)
    judge = ProcessJudge(model_config, max_tokens=max_tokens)
    counts = {"completed": 0, "invalid": 0, "missing": 0}
    try:
        rows = store.rows(experiment_id)
        for row in rows:
            if row.get("status") != "completed":
                continue
            existing = store.connection.execute(
                "SELECT status FROM process_grade WHERE run_id = ?", (row["run_id"],)
            ).fetchone()
            if existing is not None and existing[0] == "completed":
                counts["completed"] += 1
                continue
            trace_path = Path(row.get("trace_path") or "")
            if not trace_path.is_file():
                store.add_process_grade(row["run_id"], {"status": "missing", "error": "trace missing"})
                counts["missing"] += 1
                continue
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            task_source = task_root / row["task_id"] / "source"
            prompt_path = task_source / "prompt.txt"
            prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
            grade_row = store.connection.execute(
                "SELECT score, details FROM grade WHERE run_id = ? AND kind = 'task' LIMIT 1",
                (row["run_id"],),
            ).fetchone()
            outcome = float(grade_row[0]) if grade_row and grade_row[0] is not None else None
            try:
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
            store.add_process_grade(row["run_id"], result)
            counts[result.get("status", "invalid")] = counts.get(result.get("status", "invalid"), 0) + 1
            store.commit()
    finally:
        judge.close()
        store.close()
    return counts
