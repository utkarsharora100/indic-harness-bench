from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest


class PreflightError(RuntimeError):
    """A required Phase I preflight gate failed."""


def check_docker_image(config: ExperimentConfig) -> dict[str, Any]:
    if config.sandbox.get("mode") != "docker":
        return {"mode": config.sandbox.get("mode"), "checked": False}
    try:
        import docker
    except ImportError as exc:
        raise PreflightError("Docker mode requires the docker Python package") from exc
    client = docker.from_env()
    try:
        client.ping()
        image = client.images.get(config.sandbox["image"])
        digest = None
        repo_digests = getattr(image, "attrs", {}).get("RepoDigests", [])
        if repo_digests:
            digest = repo_digests[0]
        return {"mode": "docker", "image": config.sandbox["image"], "digest": digest}
    except docker.errors.DockerException as exc:
        raise PreflightError(
            f"Docker engine/image is not ready for {config.sandbox['image']}: {exc}"
        ) from exc
    finally:
        client.close()


def runtime_preflight(config: ExperimentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"experiment_id": config.experiment_id}
    for model_name in config.experiment.get("models", []):
        if model_name == "university_gpu":
            _, manifest = ensure_model_manifest(config.root, config.inference, verify_tool=True)
            result["university_gpu"] = {
                # The manifest intentionally contains the exact served ID in
                # ignored local storage.  Do not echo the server filesystem
                # path into console output, CI logs, or tracked artifacts.
                "manifest": config.inference.get(
                    "manifest", "data/phase1/model-manifest.json"
                ),
                "served_model_count": len(manifest.get("served_models", [])),
                "tool_call_verified": manifest["tool_call_check"].get("verified", False),
            }
    result["docker"] = check_docker_image(config)
    return result
