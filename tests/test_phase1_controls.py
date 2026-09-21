import json
from pathlib import Path

from agents.base import AgentRequest
from agents.react import ReactAgent
from agents.tools import WorkspaceTools
from analysis.metrics import paired_outcome
from analysis.phase1 import paired_language_deltas, paired_outcomes
from benchmark.upstream import after_round_runtime, prepare_runtime, render_runtime_template
from runner.redaction import redact
from runner.runner import stable_cell_id
from runner.log import RunStore


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


def test_workspace_mount_path_is_equivalent_to_relative_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceTools(workspace, timeout_seconds=5)
    assert tools.execute("write_file", {"path": "/workspace/out.txt", "content": "ok"})["ok"]
    assert tools.execute("read_file", {"path": "out.txt"})["content"] == "ok"


def test_upstream_hook_variables_are_rendered(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "hooks.py").write_text(
        "def prepare_runtime(context):\n"
        "    return {'TASK_ID': 'T-1'}\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = prepare_runtime(task, tmp_path, workspace, "hooks.py")
    assert state == {"TASK_ID": "T-1"}
    assert render_runtime_template("$WORKSPACE/$TASK_ID", workspace="/workspace", runtime_env=state) == "/workspace/T-1"


def test_upstream_after_round_hook_receives_task_context(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "hooks.py").write_text(
        "def after_round(context, state, result):\n"
        "    assert context['task'].task_dir.name == 'task'\n"
        "    assert context['round_index'] == 0\n"
        "    assert result.ok\n"
        "    return {'AFTER': 'yes'}\n",
        encoding="utf-8",
    )
    state = after_round_runtime(
        task,
        tmp_path,
        tmp_path / "workspace",
        "hooks.py",
        {"BEFORE": "yes"},
        type("Result", (), {"ok": True})(),
    )
    assert state == {"BEFORE": "yes", "AFTER": "yes"}


def test_all_fail_is_not_reported_as_mixed():
    assert paired_outcome(False, False, False) == "all_fail"


def test_full_grade_details_are_persisted(tmp_path: Path):
    store = RunStore(tmp_path / "runs.sqlite")
    try:
        store.add_grade("run", "task_grader", False, "x" * 20000, score=0.5)
        row = store.connection.execute("SELECT length(details), score FROM grade WHERE run_id='run'").fetchone()
        assert row[0] == 20000
        assert row[1] == 0.5
    finally:
        store.close()
