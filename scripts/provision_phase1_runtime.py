from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE_COMMITS = {
    "nanobot": ("v0.3.5", "1bb712d3488915ca4ed9ccc1a93067ff722f5ab9"),
    "openclaw": ("v2026.9.5", "ec9c1a13db8938e5a3eaa51fca2e981cde2395a9"),
}


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


def verify_release_tag(repository: str, tag: str, expected_commit: str) -> None:
    rows = run(["git", "ls-remote", repository,
                f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"])
    refs = dict(line.split("\t", 1)[::-1] for line in rows.splitlines() if "\t" in line)
    actual = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if actual != expected_commit:
        raise RuntimeError(f"The upstream {tag} release tag no longer matches its frozen source commit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and pin the Phase I native harness runtimes")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.corrected.pilot-v13.yaml"))
    parser.add_argument("--network", default="phase1-agent-net")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    agents = yaml.safe_load((ROOT / config["experiment"]["agents_manifest"]).read_text(encoding="utf-8"))["agents"]
    runtime_manifest = Path(config["inference"]["runtime_manifest"])

    images = {
        "task_tools": {
            "name": "indic-harness-task-tools:pilot-v13",
            "dockerfile": "docker/task-tools.Dockerfile",
            "source": "python:3.12-slim-bookworm",
        },
        "benchmark": {
            "name": config["sandbox"]["image"],
            "dockerfile": "docker/phase1.Dockerfile",
            "source": "indic-harness-task-tools:pilot-v13",
        },
        "nanobot": {
            "name": agents["nanobot"]["image"],
            "dockerfile": "docker/nanobot.Dockerfile",
            "release": "0.3.5",
            "source": "https://github.com/HKUDS/nanobot",
        },
        "openclaw": {
            "name": agents["openclaw"]["image"],
            "dockerfile": "docker/openclaw.Dockerfile",
            "release": "2026.9.5",
            "source": "https://github.com/openclaw/openclaw",
        },
        "proxy": {
            "name": config["inference"]["proxy"]["image"],
            "dockerfile": "docker/proxy.Dockerfile",
            "release": "phase1-pinned",
            "source": "indic-harness-bench/runner/proxy.py",
        },
    }
    run(["docker", "version", "--format", "{{.Server.Version}}"])
    for name, (tag, revision) in RELEASE_COMMITS.items():
        verify_release_tag(images[name]["source"], tag, revision)
        images[name]["source_revision"] = revision
    if not args.skip_build:
        for record in images.values():
            run(["docker", "build", "-f", record["dockerfile"], "-t", record["name"], "."])
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
    output = ROOT / runtime_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
