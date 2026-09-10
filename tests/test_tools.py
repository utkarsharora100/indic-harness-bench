from pathlib import Path

from agents.tools import WorkspaceTools


def test_workspace_write(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceTools(workspace, timeout_seconds=5)

    result = tools.execute("write_file", {"path": "hello.txt", "content": "hello"})

    assert result["ok"]
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_workspace_rejects_parent_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceTools(workspace, timeout_seconds=5)

    try:
        tools.execute("read_file", {"path": "../secret.txt"})
    except ValueError:
        return

    raise AssertionError("Parent path should be rejected")
