"""Small secret-free atomic heartbeat helpers for unattended study stages."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_heartbeat(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"updated_at": utc_now(), **fields}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_journal(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"at": utc_now(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
