import json
from pathlib import Path

from agents.base import AgentRequest
from agents.react import ReactAgent
from agents.tools import WorkspaceTools
from analysis.phase1 import paired_language_deltas, paired_outcomes
from runner.redaction import redact
from runner.runner import stable_cell_id


class _MalformedCompletions:
    def create(self, **kwargs):
        raise RuntimeError(
            "Error code: 500 - Failed to parse tool call arguments as JSON: parse_error"
        )


class _MalformedClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _MalformedCompletions()})()

    def close(self):
        return None


def test_malformed_tool_call_is_recorded_without_losing_the_run(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = object.__new__(ReactAgent)
    agent.client = _MalformedClient()

    response = agent.run(
        AgentRequest(
            instruction="Do the task.",
            workspace=str(workspace),
            system_prompt="Use tools.",
            model="test",
            temperature=0.0,
            top_p=1.0,
            max_tokens=100,
            max_steps=3,
        )
    )

    assert not response.completed
    assert response.metadata["stop_reason"] == "malformed_tool_call"
    assert response.metadata["events"][0]["event_type"] == "model_error"


def test_failed_command_result_is_observable_and_countable(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceTools(workspace, timeout_seconds=5, command_runner=lambda command, timeout: {
        "ok": False,
        "returncode": 17,
        "stdout": "",
        "stderr": "expected failure",
    })

    result = tools.execute("run_command", {"command": "false"})

    assert result["ok"] is False
    assert result["returncode"] == 17


def test_stable_cell_id_and_secret_redaction():
    first = stable_cell_id("exp", "rev", "task", "hindi", "react", "model", 2)
    second = stable_cell_id("exp", "rev", "task", "hindi", "react", "model", 2)
    assert first == second
    assert len(first) == 32
    assert redact({"key": "secret", "nested": ["secret"]}, ("secret",)) == {
        "key": "[REDACTED]",
        "nested": ["[REDACTED]"],
    }


def test_paired_report_calculations():
    rows = [
        {"task_id": "a", "repetition": 0, "language": "english", "success": 1},
        {"task_id": "a", "repetition": 0, "language": "hindi", "success": 0},
        {"task_id": "a", "repetition": 0, "language": "hinglish", "success": 1},
        {"task_id": "b", "repetition": 0, "language": "english", "success": 0},
        {"task_id": "b", "repetition": 0, "language": "hindi", "success": 1},
        {"task_id": "b", "repetition": 0, "language": "hinglish", "success": 1},
    ]
    assert paired_language_deltas(rows) == {"hindi_minus_english": 0.0, "hinglish_minus_english": 0.5}
    assert paired_outcomes(rows) == {"english_only_failure": 1, "hindi_only_failure": 1}


def test_no_json_trace_like_secret_is_required_for_redaction():
    payload = {"api_key": "sk-private", "nested": {"url": "http://private"}}
    assert json.dumps(redact(payload, ("sk-private", "http://private"))) == (
        '{"api_key": "[REDACTED]", "nested": {"url": "[REDACTED]"}}'
    )
