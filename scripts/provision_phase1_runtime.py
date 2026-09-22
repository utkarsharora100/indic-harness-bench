from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def image_id(name: str) -> str:
    value = run(["docker", "image", "inspect", name, "--format", "{{.Id}}"])
    if not value.startswith("sha256:"):
        raise RuntimeError(f"Docker did not return an immutable image ID for {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and pin the Phase I native harness runtimes")
    parser.add_argument("--runtime-manifest", type=Path, default=Path("data/phase1/corrected/runtime-manifest.json"))
    parser.add_argument("--network", default="phase1-agent-net")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    images = {
        "nanobot": {
            "name": "indic-harness-nanobot:phase1-pinned",
            "dockerfile": "docker/nanobot.Dockerfile",
            "release": "0.3.5",
            "source": "https://github.com/HKUDS/nanobot",
        },
        "openclaw": {
            "name": "indic-harness-openclaw:phase1-pinned",
            "dockerfile": "docker/openclaw.Dockerfile",
            "release": "2026.9.5",
            "source": "https://github.com/openclaw/openclaw",
        },
        "proxy": {
            "name": "indic-harness-proxy:phase1-pinned",
            "dockerfile": "docker/proxy.Dockerfile",
            "release": "phase1-pinned",
            "source": "indic-harness-bench/runner/proxy.py",
        },
    }
    run(["docker", "version", "--format", "{{.Server.Version}}"])
    if not args.skip_build:
        for record in images.values():
            run(["docker", "build", "--pull", "-f", record["dockerfile"], "-t", record["name"], "."])
    try:
        run(["docker", "network", "inspect", args.network])
    except RuntimeError:
        run(["docker", "network", "create", "--internal", args.network])
    network_info = json.loads(run(["docker", "network", "inspect", args.network]))[0]
    if not network_info.get("Internal"):
        raise RuntimeError(f"Network {args.network} is not Docker-internal")
    for record in images.values():
        record["image_id"] = image_id(record["name"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "network": {"name": args.network, "internal": True},
        "images": images,
    }
    output = ROOT / args.runtime_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
