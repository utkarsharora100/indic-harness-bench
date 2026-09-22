from pathlib import Path

from agents.react import ReactAgent


class FakeCall:
    id = "call-1"

    class function:
        name = "read_file"
        arguments = '{"path": "app.py"}'


class FakeMessage:
    content = None
    tool_calls = [FakeCall()]


class FakeChoice:
    message = FakeMessage()


class FakeUsage:
    prompt_tokens = 4
    completion_tokens = 3


class FakeCompletion:
    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return FakeCompletion()

        completion = FakeCompletion()
        completion.choices = [type("Choice", (), {
            "message": type("Message", (), {"content": "done", "tool_calls": []})()
        })()]
        return completion


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    def close(self):
        pass


def test_react_records_tool_calls(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('ok')\n", encoding="utf-8")

    agent = object.__new__(ReactAgent)
    agent.client = FakeClient()

    from agents.base import AgentRequest

    result = agent.run(AgentRequest(
        instruction="Read app.py.",
        workspace=str(workspace),
        system_prompt="Use tools.",
        model="test",
        temperature=0.0,
        top_p=1.0,
        max_tokens=100,
        max_steps=3,
    ))

    assert result.completed
    assert result.metadata["tool_calls"] == 1
    assert result.metadata["failed_tool_calls"] == 0
    assert result.metadata["events"][0]["tool"] == "read_file"


def test_react_stops_after_truncated_tool_call_without_continuation(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('ok')\n", encoding="utf-8")

    class LengthChoice:
        message = FakeMessage()
        finish_reason = "length"

    class LengthCompletion:
        choices = [LengthChoice()]
        usage = FakeUsage()

    class LengthCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return LengthCompletion()

    agent = object.__new__(ReactAgent)
    completions = LengthCompletions()
    agent.client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})(), "close": lambda self: None},
    )()

    from agents.base import AgentRequest

    result = agent.run(AgentRequest(
        instruction="Read app.py.",
        workspace=str(workspace),
        system_prompt="Use tools.",
        model="test",
        temperature=0.0,
        top_p=1.0,
        max_tokens=100,
        max_steps=3,
    ))

    assert not result.completed
    assert result.metadata["stop_reason"] == "max_tokens"
    assert result.metadata["tool_calls"] == 1
    assert completions.calls == 1
