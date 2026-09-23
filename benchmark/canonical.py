"""Read task assets from a pinned Git tree without checkout newline conversion."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_bytes(source: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Git object read failed: {' '.join(args)}: "
            + result.stderr.decode("utf-8", errors="replace")[-500:]
        )
    return result.stdout


def task_blobs(source: Path, revision: str, task_id: str) -> dict[str, tuple[bytes, bool]]:
    prefix = f"tasks/{task_id}/"
    listing = git_bytes(source, "ls-tree", "-r", "-z", revision, "--", f"tasks/{task_id}")
    blobs: dict[str, tuple[bytes, bool]] = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if kind != "blob" or mode not in {"100644", "100755"} or not path.startswith(prefix):
            raise ValueError(f"Unsupported pinned task asset: {path} ({mode} {kind})")
        relative = path[len(prefix):]
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe pinned task path: {path}")
        blobs[relative] = (git_bytes(source, "cat-file", "blob", object_id), mode == "100755")
    if not blobs:
        raise FileNotFoundError(f"No files in pinned task: {task_id}")
    return blobs


def blob_tree_sha256(blobs: dict[str, tuple[bytes, bool]]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(blobs):
        if "__pycache__" in Path(relative).parts or Path(relative).suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(blobs[relative][0])
        digest.update(b"\0")
    return digest.hexdigest()


def materialize_blobs(blobs: dict[str, tuple[bytes, bool]], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace pinned assets: {destination}")
    destination.mkdir(parents=True)
    for relative, (data, executable) in blobs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if executable:
            target.chmod(target.stat().st_mode | 0o111)
