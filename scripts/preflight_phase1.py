from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.loader import select_tasks
from runner.config import ExperimentConfig
from runner.inference import InferenceEndpoint, verify_tool_call
from runner.preflight import runtime_preflight
from runner.runner import ExperimentRunner
from scripts.prepare_phase1 import prepare
from runner.cli import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase I dataset, matrix, model, and Docker preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Pinned Harness-Bench checkout")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    result: dict[str, object] = {}
    if args.source:
        result["dataset"] = prepare(
            args.source.resolve(),
            config.root / "benchmark/task_selection.yaml",
            config.root / "benchmark/translations/phase1.yaml",
            config.task_root,
            check_only=True,
        )
    agents, models, _ = load_runtime(config)
    tasks = select_tasks(config.task_root, config.configured_task_ids)
    if int(config.generation.get("max_steps", 40)) != 40:
        raise RuntimeError("Corrective Phase I requires a 40-step maximum")
    if float(config.generation.get("temperature", 0)) != 0.0 or float(config.generation.get("top_p", 1)) != 1.0:
        raise RuntimeError("Corrective Phase I requires temperature=0 and top_p=1")
    for task in tasks:
        if task.upstream is not None:
            source = config.task_root / task.task_id / task.upstream.source_dir
            hook = source / task.upstream.hooks_module
            if hook.exists() and not hook.is_file():
                raise RuntimeError(f"Task hook is not a file: {hook}")
            if task.limits.max_steps > int(config.generation.get("max_steps", 40)):
                raise RuntimeError(f"Task {task.task_id} exceeds configured step budget")
            if task.upstream.prompt_files:
                raise RuntimeError(
                    f"Task {task.task_id} declares multi-round prompts; corrective overlays do not yet cover them"
                )
    runner = ExperimentRunner(config, agents, models)
    try:
        result["matrix"] = runner.validate_matrix(tasks)
        if config.inference.get("proxy", {}).get("enabled"):
            model_name = str(config.experiment["models"][0])
            model = runner.models[model_name]
            endpoint = InferenceEndpoint(
                base_url=str(model["base_url"]),
                api_key=str(model["api_key"]),
                requested_model=None,
                resolved_model=str(model["model"]),
                served_models=(str(model["model"]),),
            )
            result["proxy"] = {
                "tool_call_verified": verify_tool_call(
                    endpoint, int(config.inference.get("timeout_seconds", 90))
                ).get("verified", False),
                "direct_internet_blocked_by_internal_network": True,
            }
        for agent_name in config.experiment.get("agents", []):
            agent = agents.get(agent_name, {})
            if agent.get("type") != "native":
                continue
            try:
                import docker

                client = docker.from_env()
                try:
                    client.ping()
                    image = client.images.get(agent["image"])
                    expected_digest = str(agent.get("image_digest", "")).strip()
                    if not expected_digest:
                        raise RuntimeError(
                            "native harness image_digest is not frozen; record the exact image digest before the pilot"
                        )
                    image_id = str(image.attrs.get("Id", ""))
                    repo_digests = [str(value) for value in image.attrs.get("RepoDigests", [])]
                    if expected_digest not in image_id and not any(
                        expected_digest in value for value in repo_digests
                    ):
                        raise RuntimeError(
                            f"installed image digest does not match frozen digest {expected_digest}"
                        )
                    network = str(agent.get("network", ""))
                    if not network or network == "none":
                        raise RuntimeError("native harness must use the internal model-proxy network")
                    network_info = client.networks.get(network)
                    if not bool(network_info.attrs.get("Internal")):
                        raise RuntimeError(
                            f"Native harness network {network} must be Docker-internal"
                        )
                finally:
                    client.close()
            except Exception as exc:
                raise RuntimeError(
                    f"Native harness {agent_name} is not installed as pinned image {agent.get('image')}: {exc}"
                ) from exc
    finally:
        runner.close()
    result["runtime"] = runtime_preflight(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
