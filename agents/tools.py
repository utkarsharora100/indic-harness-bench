from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories below a path inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path; use '.' for root."}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a file inside the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search UTF-8 text files below a directory for a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command with the workspace as the current directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


class WorkspaceTools:
    def __init__(self, workspace: Path, timeout_seconds: int, command_runner=None) -> None:
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner

    def schemas(self) -> list[dict[str, Any]]:
        return TOOLS

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "run_command": self._run_command,
        }
        try:
            handler = handlers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        return handler(**arguments)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.workspace / relative_path).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError(f"Path is outside workspace: {relative_path}")
        return path

    def _list_files(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            return {"ok": False, "error": "path does not exist"}
        return {
            "ok": True,
            "entries": [
                {"name": item.name, "type": "directory" if item.is_dir() else "file"}
                for item in sorted(target.iterdir())
            ],
        }

    def _read_file(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            return {"ok": False, "error": "file does not exist"}
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": "file is not UTF-8 text"}
        return {"ok": True, "content": content}

    def _write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}

    def _search_files(self, query: str, path: str) -> dict[str, Any]:
        root = self._resolve(path)
        if not root.exists():
            return {"ok": False, "error": "path does not exist"}
        matches = []
        for target in root.rglob("*"):
            if not target.is_file():
                continue
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if query in text:
                matches.append(str(target.relative_to(self.workspace)))
        return {"ok": True, "matches": matches}

    def _run_command(self, command: str) -> dict[str, Any]:
        if self.command_runner is not None:
            return self.command_runner(command)

        try:
            process = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=os.environ.copy(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "timeout": True,
                "stdout": (exc.stdout or "")[-12000:],
                "stderr": (exc.stderr or "")[-12000:],
            }
        return {
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-12000:],
            "stderr": process.stderr[-12000:],
        }


def serialise_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
