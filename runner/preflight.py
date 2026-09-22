from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import yaml

from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest
from runner.inference import InferenceEndpoint, verify_tool_call
from runner.proxy import InferenceProxy
from runner.proxy_sidecar import ProxySidecar
from benchmark.loader import select_tasks


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


def _runtime_manifest(config: ExperimentConfig) -> dict[str, Any]:
    path = config.root / config.inference.get(
        "runtime_manifest", "data/phase1/corrected/runtime-manifest.json"
    )
    if not path.is_file():
        raise PreflightError(f"Native runtime manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Native runtime manifest is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError("Native runtime manifest must be a JSON object")
    return value


def check_native_runtimes(config: ExperimentConfig, agents: dict[str, Any]) -> dict[str, Any]:
    native = {name: value for name, value in agents.items() if value.get("type") == "native"}
    if not native:
        return {"native": {}}
    try:
        import docker
    except ImportError as exc:
        raise PreflightError("Native runtimes require the Docker Python package") from exc

    manifest = _runtime_manifest(config)
    manifest_images = manifest.get("images") or {}
    client = docker.from_env()
    checked: dict[str, Any] = {}
    try:
        client.ping()
        for name, agent in native.items():
            image_name = str(agent.get("image", ""))
            if not image_name:
                raise PreflightError(f"Native runtime {name} has no image")
            try:
                image = client.images.get(image_name)
            except docker.errors.DockerException as exc:
                raise PreflightError(f"Native runtime image is unavailable for {name}: {image_name}") from exc
            image_id = str(image.attrs.get("Id", ""))
            expected = str(agent.get("image_digest") or manifest_images.get(name, {}).get("image_id", ""))
            if not expected or expected not in image_id:
                raise PreflightError(f"Native runtime {name} is not pinned to its recorded image ID")
            network_name = str(agent.get("network", ""))
            if not network_name:
                raise PreflightError(f"Native runtime {name} has no isolated network")
            try:
                network = client.networks.get(network_name)
            except docker.errors.DockerException as exc:
                raise PreflightError(f"Native runtime network is unavailable: {network_name}") from exc
            if not bool(network.attrs.get("Internal")):
                raise PreflightError(f"Native runtime network {network_name} must be Docker-internal")

            version_command = ["nanobot", "--version"] if name == "nanobot" else ["openclaw", "--version"]
            def run_probe(command: list[str]) -> int:
                container = client.containers.create(
                    image_name,
                    command=command,
                    network=network_name,
                    network_disabled=False,
                    extra_hosts={"host.docker.internal": "host-gateway"},
                )
                try:
                    container.start()
                    result = container.wait(timeout=30)
                    return int(result.get("StatusCode", 1)) if isinstance(result, dict) else 1
                finally:
                    container.remove(force=True)

            if run_probe(version_command) != 0:
                raise PreflightError(f"Native runtime {name} failed its version/startup check")
            # An internal network must not provide direct public egress. The
            # command is expected to fail; success is a hard gate failure.
            probe_status = run_probe(
                [
                    "python3",
                    "-c",
                    "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3)",
                ]
            )
            if probe_status == 0:
                raise PreflightError(f"Native runtime {name} can reach the public internet")
            checked[name] = {"image_id": image_id, "network": network_name, "startup_verified": True}
    except docker.errors.DockerException as exc:
        raise PreflightError(f"Native runtime Docker preflight failed: {exc}") from exc
    finally:
        client.close()
    return {"native": checked}


def check_proxy_image(config: ExperimentConfig) -> dict[str, Any]:
    """Verify that the sidecar image is present and matches the provision record."""
    proxy_config = config.inference.get("proxy") or {}
    if not proxy_config.get("enabled"):
        return {"enabled": False}
    image_name = str(proxy_config.get("image", ""))
    if not image_name:
        raise PreflightError("The native-harness model proxy has no image")
    manifest = _runtime_manifest(config)
    expected = str((manifest.get("images") or {}).get("proxy", {}).get("image_id", ""))
    try:
        import docker
    except ImportError as exc:
        raise PreflightError("The model proxy requires the Docker Python package") from exc
    client = docker.from_env()
    try:
        image = client.images.get(image_name)
        image_id = str(image.attrs.get("Id", ""))
        if not expected or expected not in image_id:
            raise PreflightError("The model proxy image is not pinned to its provisioned image ID")
        return {"enabled": True, "image": image_name, "image_id": image_id}
    except docker.errors.DockerException as exc:
        raise PreflightError(f"Model proxy image is unavailable: {image_name}") from exc
    finally:
        client.close()


def check_proxy_sidecar(config: ExperimentConfig, agents: dict[str, Any]) -> dict[str, Any]:
    """Exercise the actual internal-network sidecar before any cells are written."""
    proxy_config = config.inference.get("proxy") or {}
    if not proxy_config.get("enabled"):
        return {"tool_call_verified": False, "enabled": False}
    native = [value for value in agents.values() if value.get("type") == "native"]
    if not native:
        raise PreflightError("The proxy sidecar requires at least one native harness")
    try:
        import docker
    except ImportError as exc:
        raise PreflightError("The model proxy requires the Docker Python package") from exc

    endpoint, _manifest = ensure_model_manifest(config.root, config.inference, verify_tool=True)
    public_model = str(proxy_config.get("public_model", "phase1-university-model"))
    internal_network = str(native[0].get("network", ""))
    trace_root = config.root / config.storage.get("traces_dir", "data/phase1/traces") / "preflight-sidecar"
    sidecar: ProxySidecar | None = None
    client = docker.from_env()
    probe = None
    try:
        sidecar = ProxySidecar(
            image=str(proxy_config.get("image")),
            internal_network=internal_network,
            egress_network=str(proxy_config.get("egress_network", "bridge")),
            upstream_base_url=endpoint.base_url,
            upstream_key=endpoint.api_key,
            resolved_model=endpoint.resolved_model,
            public_model=public_model,
            trace_root=trace_root,
            cell_id="preflight-proxy",
            max_calls=int(proxy_config.get("max_calls_per_cell", 40)),
            temperature=float(config.generation.get("temperature", 0.0)),
            top_p=float(config.generation.get("top_p", 1.0)),
            max_tokens=int(config.generation.get("max_tokens", 2048)),
        ).start()
        # The sidecar image starts a Python HTTP server; allow its process to
        # finish booting before the disposable probe joins the internal net.
        import time

        time.sleep(1.0)
        probe_script = """
import json, os, urllib.request
base = os.environ['PHASE1_PROXY_URL'].rstrip('/')
headers = {'Authorization': 'Bearer ' + os.environ['PHASE1_PROXY_KEY'], 'Content-Type': 'application/json'}
def call(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=body, headers=headers, method='POST' if body else 'GET')
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    except Exception as exc:
        print('probe-call-failed:' + path + ':' + type(exc).__name__ + ':' + str(getattr(exc, 'code', '')))
        raise
models = call('/models')
ids = [item.get('id') for item in models.get('data', []) if isinstance(item, dict)]
if ids != [os.environ['PHASE1_PROXY_MODEL']]:
    raise RuntimeError('models-mismatch')
result = call('/chat/completions', {
    'model': os.environ['PHASE1_PROXY_MODEL'],
    'messages': [{'role': 'user', 'content': 'Use the add tool to add 2 and 3.'}],
    'tools': [{'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}],
    'tool_choice': 'auto', 'temperature': 0, 'top_p': 1, 'max_tokens': 2048, 'stream': False,
})
calls = ((result.get('choices') or [{}])[0].get('message') or {}).get('tool_calls')
if not calls or calls[0].get('function', {}).get('name') != 'add':
    raise RuntimeError('tool-call-mismatch')
print('proxy-ok')
"""
        probe = client.containers.run(
            config.sandbox["image"],
            command=["python", "-c", probe_script],
            network=internal_network,
            environment={
                "PHASE1_PROXY_URL": sidecar.base_url,
                "PHASE1_PROXY_KEY": sidecar.client_key,
                "PHASE1_PROXY_MODEL": public_model,
            },
            detach=True,
        )
        status = probe.wait(timeout=60)
        output = probe.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        status_code = int(status.get("StatusCode", 1)) if isinstance(status, dict) else 1
        if status_code != 0 or "proxy-ok" not in output:
            diagnostic = ";".join(
                line.strip()
                for line in output.splitlines()
                if line.strip().startswith("probe-call-failed:")
                or line.strip().endswith("models-mismatch")
                or line.strip().endswith("tool-call-mismatch")
            )
            raise PreflightError(
                f"The internal model-proxy tool-call probe failed (exit={status_code}, {diagnostic or 'no-safe-diagnostic'})"
            )
        snapshot = sidecar.snapshot()
        if int(snapshot.get("calls", 0)) != 1:
            raise PreflightError("The internal model proxy did not record the probe call")
        if endpoint.resolved_model in json.dumps(snapshot, ensure_ascii=False):
            raise PreflightError("The private served model identifier appeared in sidecar trace data")
        return {"enabled": True, "tool_call_verified": True, "calls": 1}
    except (docker.errors.DockerException, OSError, TimeoutError) as exc:
        raise PreflightError("The internal model-proxy sidecar preflight failed") from exc
    finally:
        if probe is not None:
            try:
                probe.remove(force=True)
            except Exception:
                pass
        if sidecar is not None:
            sidecar.close()
        client.close()


def check_prepared_tasks(config: ExperimentConfig, tasks: list[Any]) -> dict[str, Any]:
    selection_path = config.root / config.experiment.get("dataset_manifest", "benchmark/task_selection.yaml")
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    expected = {str(item["task_id"]): str(item.get("source_sha256", "")) for item in selection.get("tasks", [])}
    if set(expected) != {task.task_id for task in tasks}:
        raise PreflightError("Prepared task set does not match the pinned selection manifest")
    checked = 0
    for task in tasks:
        task_root = config.task_root / task.task_id
        marker = task_root / ".source_sha256"
        source = task_root / "source"
        if not marker.is_file() or not source.is_dir():
            raise PreflightError(f"Prepared task is incomplete: {task.task_id}")
        actual = marker.read_text(encoding="utf-8").strip()
        if expected.get(task.task_id) and actual != expected[task.task_id]:
            raise PreflightError(f"Prepared task hash mismatch: {task.task_id}")
        for language in ("english", "hindi", "hinglish"):
            prompt = task.instruction_for(language)
            if not isinstance(prompt, str) or not prompt.strip():
                raise PreflightError(f"Empty {language} prompt: {task.task_id}")
        checked += 1
    return {"prepared_tasks": checked, "language_variants": checked * 3}


def full_preflight(
    config: ExperimentConfig,
    tasks: list[Any],
    agents: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    matrix = {
        "tasks": len(tasks),
        "languages": len(config.experiment.get("languages", [])),
        "agents": len(config.experiment.get("agents", [])),
        "repetitions": int(config.experiment.get("repetitions", 0)),
    }
    matrix["cells"] = matrix["tasks"] * matrix["languages"] * matrix["agents"] * matrix["repetitions"]
    if matrix["cells"] not in {45, 648}:
        raise PreflightError(f"Unexpected corrective Phase I matrix size: {matrix['cells']}")
    result = {"experiment_id": config.experiment_id, "matrix": matrix}
    selection_path = config.root / config.experiment.get("dataset_manifest", "benchmark/task_selection.yaml")
    selection_data = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    all_ids = [str(item["task_id"]) for item in selection_data.get("tasks", [])]
    all_prepared = select_tasks(config.task_root, all_ids)
    configured_ids = {task.task_id for task in tasks}
    if not configured_ids.issubset({task.task_id for task in all_prepared}):
        raise PreflightError("Configured pilot tasks are not all present in the prepared cache")
    result.update(check_prepared_tasks(config, all_prepared))
    result["docker"] = check_docker_image(config)
    result.update(check_native_runtimes(config, agents))
    result["proxy_image"] = check_proxy_image(config)
    result["proxy_sidecar"] = check_proxy_sidecar(config, agents)
    result["runtime"] = runtime_preflight(config)
    return result


def runtime_preflight(config: ExperimentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"experiment_id": config.experiment_id}
    for model_name in config.experiment.get("models", []):
        if model_name == "university_gpu":
            endpoint, manifest = ensure_model_manifest(config.root, config.inference, verify_tool=True)
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
            proxy_config = config.inference.get("proxy") or {}
            if proxy_config.get("enabled"):
                proxy = InferenceProxy(
                    endpoint.base_url,
                    endpoint.api_key,
                    endpoint.resolved_model,
                    public_model=str(proxy_config.get("public_model", "phase1-university-model")),
                    max_calls_per_cell=int(proxy_config.get("max_calls_per_cell", 40)),
                    temperature=float(config.generation.get("temperature", 0.0)),
                    top_p=float(config.generation.get("top_p", 1.0)),
                    max_tokens=int(config.generation.get("max_tokens", 2048)),
                ).start()
                try:
                    proxy_endpoint = InferenceEndpoint(
                        proxy.base_url,
                        proxy.client_key,
                        None,
                        str(proxy_config.get("public_model", "phase1-university-model")),
                        (str(proxy_config.get("public_model", "phase1-university-model")),),
                    )
                    proxy.begin_cell("preflight")
                    tool_check = verify_tool_call(
                        proxy_endpoint,
                        int(config.inference.get("timeout_seconds", 90)),
                        max_tokens=int(config.generation.get("max_tokens", 2048)),
                    )
                    events = proxy.end_cell()
                    if any(endpoint.resolved_model in json.dumps(event, ensure_ascii=False) for event in events):
                        raise RuntimeError("Private served model identifier appeared in proxy trace")
                    result["proxy"] = {"tool_call_verified": tool_check.get("verified", False), "events": len(events)}
                finally:
                    proxy.close()
    result["docker"] = check_docker_image(config)
    return result
