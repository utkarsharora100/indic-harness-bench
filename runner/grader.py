from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic


@dataclass(slots=True)
class GradeResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def run_grader(
    workspace: Path,
    command: str,
    workdir: str,
    expected_exit_code: int,
    timeout_seconds: int,
    command_runner=None,
) -> GradeResult:
    started = monotonic()
    if command_runner is not None:
        result = command_runner(command)
        return GradeResult(
            success=result.get("returncode") == expected_exit_code,
            returncode=int(result.get("returncode", 1)),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            elapsed_seconds=monotonic() - started,
        )

    process = subprocess.run(
        command,
        cwd=(workspace / workdir).resolve(),
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return GradeResult(
        success=process.returncode == expected_exit_code,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        elapsed_seconds=monotonic() - started,
    )
