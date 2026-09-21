from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from benchmark.upstream import run_oracle


class GraderInfrastructureError(RuntimeError):
    """The grader could not produce a trustworthy result."""


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
    command_runner: Callable[[str, int], dict[str, Any]] | None = None,
) -> GradeResult:
    started = monotonic()
    if command_runner is not None:
        result = command_runner(command, timeout_seconds)
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


def run_upstream_oracle(
    task_dir: Path,
    oracle_module: str,
    workspace: Path,
    expected_outcome_score: float,
    *,
    oracle_runner: Callable[[Path, str, int], dict[str, Any]] | None = None,
    timeout_seconds: int = 600,
) -> GradeResult:
    started = monotonic()
    if oracle_runner is None:
        result = run_oracle(task_dir, oracle_module, workspace)
    else:
        result = oracle_runner(task_dir, oracle_module, timeout_seconds)
    score = result.get("outcome_score", 0.0)
    try:
        success = float(score) >= expected_outcome_score
    except (TypeError, ValueError):
        success = False
    payload = json.dumps(result, ensure_ascii=False)
    return GradeResult(
        success=success,
        returncode=0 if success else 1,
        stdout=payload,
        stderr=str(result.get("error", "")),
        elapsed_seconds=monotonic() - started,
    )
