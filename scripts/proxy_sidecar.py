from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from runner.proxy import InferenceProxy


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing sidecar setting: {name}")
    return value


def main() -> None:
    trace_dir = Path(os.environ.get("PHASE1_TRACE_DIR", "/trace"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    proxy = InferenceProxy(
        required("PHASE1_UPSTREAM_BASE_URL"),
        required("PHASE1_UPSTREAM_API_KEY"),
        required("PHASE1_RESOLVED_MODEL"),
        public_model=os.environ.get("PHASE1_PUBLIC_MODEL", "phase1-university-model"),
        max_calls_per_cell=int(os.environ.get("PHASE1_MAX_CALLS", "40")),
        temperature=float(os.environ.get("PHASE1_TEMPERATURE", "0")),
        top_p=float(os.environ.get("PHASE1_TOP_P", "1")),
        max_tokens=int(os.environ.get("PHASE1_MAX_TOKENS", "2048")),
        bind_host="0.0.0.0",
        listen_port=int(os.environ.get("PHASE1_PORT", "8080")),
        advertised_host="phase1-model-proxy",
        client_key=required("PHASE1_CLIENT_KEY"),
    ).start()
    proxy.begin_cell(required("PHASE1_CELL_ID"))
    output = trace_dir / "proxy.json"
    temporary = trace_dir / "proxy.json.tmp"
    stop = threading.Event()

    def persist() -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(proxy.snapshot_cell(), handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)

    def write_snapshot() -> None:
        while not stop.is_set():
            persist()
            stop.wait(0.25)
        persist()

    writer = threading.Thread(target=write_snapshot, daemon=True)
    writer.start()

    def shutdown(*_args: object) -> None:
        stop.set()
        proxy.close()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        while not stop.wait(1):
            pass
    finally:
        shutdown()
        writer.join(timeout=2)


if __name__ == "__main__":
    main()
