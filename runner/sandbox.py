from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmark.upstream import copy_fixtures



class WorkspaceSandbox:
    def __init__(
        self,
        source_workspace: Path | None,
        image: str,
        mode: str,
        fixtures: Path | None = None,
    ) -> None:
        self.source_workspace = source_workspace.resolve() if source_workspace else None
        self.fixtures = fixtures.resolve() if fixtures else None
        self.image = image
        self.mode = mode
        self.root: Path | None = None
        self.client: Any | None = None
        self.container: Any | None = None

    @property
    def workspace(self) -> Path:
        if self.root is None:
            raise RuntimeError("Sandbox has not been started")
        return self.root / "workspace"

    def __enter__(self) -> "WorkspaceSandbox":
        if self.source_workspace is not None and not self.source_workspace.is_dir():
            raise FileNotFoundError(f"Workspace not found: {self.source_workspace}")

        self.root = Path(tempfile.mkdtemp(prefix="ihb-"))
        if self.source_workspace is not None:
            shutil.copytree(self.source_workspace, self.workspace)
        else:
            self.workspace.mkdir()
        if self.fixtures is not None:
            copy_fixtures(self.fixtures, self.workspace)

        if self.mode == "local":
            return self

        if self.mode != "docker":
            raise ValueError(f"Unsupported sandbox mode: {self.mode}")

        try:
            import docker
        except ImportError as exc:
            self.__exit__(None, None, None)
            raise RuntimeError("Docker mode requires the docker Python package") from exc

        try:
            self.client = docker.from_env()
            self.client.ping()
            self.container = self.client.containers.run(
                self.image,
                command=["sleep", "infinity"],
                detach=True,
                working_dir="/workspace",
                volumes={
                    str(self.workspace): {"bind": "/workspace", "mode": "rw"}
                },
            )
        except docker.errors.DockerException as exc:
            self.__exit__(None, None, None)
            raise RuntimeError("Docker mode requires a running Docker daemon") from exc

        return self

    def run_command(self, command: str) -> dict[str, Any]:
        if self.container is None:
            raise RuntimeError("No container is running")

        result = self.container.exec_run(
            ["sh", "-lc", command],
            workdir="/workspace",
            demux=True,
        )
        stdout, stderr = result.output
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        return {
            "ok": result.exit_code == 0,
            "returncode": result.exit_code,
            "stdout": stdout_text[-12000:],
            "stderr": stderr_text[-12000:],
        }

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            finally:
                self.container = None

        if self.client is not None:
            self.client.close()
            self.client = None

        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None
