from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from benchmark.upstream import copy_fixtures


class SandboxInfrastructureError(RuntimeError):
    """The Docker workspace or isolated grader could not be started."""


class WorkspaceSandbox:
    def __init__(
        self,
        source_workspace: Path | None,
        image: str,
        mode: str,
        fixtures: Path | None = None,
        network: str = "none",
    ) -> None:
        self.source_workspace = source_workspace.resolve() if source_workspace else None
        self.fixtures = fixtures.resolve() if fixtures else None
        self.image = image
        self.mode = mode
        self.network = network
        self.network_disabled = network == "none"
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
            raise SandboxInfrastructureError("Docker mode requires the docker Python package") from exc

        try:
            self.client = docker.from_env()
            self.client.ping()
            self.container = self.client.containers.run(
                self.image,
                command=["sleep", "infinity"],
                detach=True,
                working_dir="/workspace",
                network_disabled=self.network_disabled,
                volumes={str(self.workspace): {"bind": "/workspace", "mode": "rw"}},
            )
        except docker.errors.DockerException as exc:
            self.__exit__(None, None, None)
            raise SandboxInfrastructureError(
                "Docker mode requires a running daemon and a usable Phase I image"
            ) from exc
        return self

    @staticmethod
    def _decode(value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def run_command(self, command: str, timeout_seconds: int | None = None) -> dict[str, Any]:
        if self.mode == "local":
            raise RuntimeError("run_command is only used as a Docker command runner")
        if self.container is None:
            raise RuntimeError("No container is running")
        seconds = int(timeout_seconds or 600)
        result = self.container.exec_run(
            ["timeout", "--signal=TERM", str(seconds), "sh", "-lc", command],
            workdir="/workspace",
            demux=True,
        )
        stdout, stderr = result.output
        return {
            "ok": result.exit_code == 0,
            "returncode": result.exit_code,
            "timeout": result.exit_code == 124,
            "stdout": self._decode(stdout)[-12000:],
            "stderr": self._decode(stderr)[-12000:],
        }

    def _ephemeral_container(
        self,
        command: list[str],
        *,
        volumes: dict[str, dict[str, str]],
        workdir: str,
        timeout_seconds: int,
        max_output_chars: int | None = 12000,
    ) -> dict[str, Any]:
        if self.client is None:
            raise SandboxInfrastructureError("Docker client is not available for isolated grading")
        container = None
        try:
            container = self.client.containers.create(
                self.image,
                command=command,
                working_dir=workdir,
                network_disabled=self.network_disabled,
                volumes=volumes,
            )
            container.start()
            waited = container.wait(timeout=max(timeout_seconds + 10, 30))
            raw_logs = container.logs(stdout=True, stderr=True)
            returncode = int(waited.get("StatusCode", 1)) if isinstance(waited, dict) else 1
            output = self._decode(raw_logs)
            if max_output_chars is not None:
                output = output[-max_output_chars:]
            return {
                "ok": returncode == 0,
                "returncode": returncode,
                "stdout": output,
                "stderr": "",
            }
        except Exception as exc:
            raise SandboxInfrastructureError(f"Isolated Docker operation failed: {exc}") from exc
        finally:
            try:
                if container is not None:
                    container.remove(force=True)
            except Exception:
                pass

    def run_grader_command(
        self,
        command: str,
        workdir: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="ihb-grader-") as temporary:
            grader_workspace = Path(temporary) / "workspace"
            shutil.copytree(self.workspace, grader_workspace)
            workdir = workdir.strip("/")
            container_workdir = "/workspace" if not workdir or workdir == "." else f"/workspace/{workdir}"
            return self._ephemeral_container(
                [
                    "timeout",
                    "--signal=TERM",
                    str(int(timeout_seconds)),
                    "sh",
                    "-lc",
                    command,
                ],
                volumes={str(grader_workspace): {"bind": "/workspace", "mode": "rw"}},
                workdir=container_workdir,
                timeout_seconds=timeout_seconds,
            )

    def run_oracle(
        self,
        task_dir: Path,
        oracle_module: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Grade a copy of the final workspace in a second container."""
        oracle_script = r"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

task_dir = Path('/grader-source')
workspace = Path('/workspace')
oracle_path = task_dir / sys.argv[1]
module_name = 'phase1_oracle'
try:
    spec = importlib.util.spec_from_file_location(module_name, oracle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('unable to load oracle module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = getattr(module, 'score_workspace', None)
    if not callable(scorer):
        raise RuntimeError('oracle missing score_workspace(workspace)')
    result = scorer(workspace)
    if not isinstance(result, dict):
        raise RuntimeError('oracle returned a non-object result')
    print(json.dumps(result, ensure_ascii=False))
except Exception as exc:
    print(json.dumps({'outcome_score': None, 'error': f'{type(exc).__name__}: {exc}',
                      'traceback': traceback.format_exc()}))
    sys.exit(2)
"""
        with tempfile.TemporaryDirectory(prefix="ihb-grader-") as temporary:
            grader_workspace = Path(temporary) / "workspace"
            shutil.copytree(self.workspace, grader_workspace)
            result = self._ephemeral_container(
                ["python", "-c", oracle_script, oracle_module],
                volumes={
                    str(grader_workspace): {"bind": "/workspace", "mode": "rw"},
                    str(task_dir.resolve()): {"bind": "/grader-source", "mode": "ro"},
                },
                workdir="/workspace",
                timeout_seconds=timeout_seconds,
                # The oracle emits one structured JSON line. Preserve the
                # complete output so a large checks list cannot truncate the
                # leading outcome_score and turn a model failure into an
                # infrastructure failure.
                max_output_chars=None,
            )
            lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
            payload = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and "outcome_score" in candidate:
                    payload = candidate
                    break
            if payload is not None:
                # The wrapper deliberately serializes oracle exceptions as a
                # score-0 payload. Such a result is gradable (the submitted
                # workspace caused the oracle to fail), even though the
                # wrapper exits non-zero so the error remains visible.
                return payload
            if result["returncode"] != 0:
                raise SandboxInfrastructureError(
                    f"Grader container exited {result['returncode']}: {result['stdout'][-1000:]}"
                )
            if not lines:
                raise SandboxInfrastructureError("Grader container returned no JSON result")
            if payload is None:
                raise SandboxInfrastructureError(
                    f"Grader container returned invalid JSON: {result['stdout'][-1000:]}"
                )

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
