from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import yaml

from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest
from runner.inference import InferenceEndpoint, verify_tool_call
from runner.proxy import InferenceProxy
from runner.proxy_sidecar import ProxySidecar
from benchmark.loader import select_tasks
from benchmark.translation import validate_translation
from benchmark.upstream import cleanup_runtime, prepare_runtime, render_runtime_template
from runner.grader import run_upstream_oracle
from runner.sandbox import WorkspaceSandbox
from scripts.prepare_phase1 import tree_sha256


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
        if config.is_corrected_phase1:
            expected = str(((_runtime_manifest(config).get("images") or {}).get("benchmark") or {}).get("image_id", ""))
            if not expected or str(image.attrs.get("Id", "")) != expected:
                raise PreflightError("Benchmark image differs from the frozen runtime manifest")
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


def check_shared_task_tools(config: ExperimentConfig, agents: dict[str, Any]) -> dict[str, Any]:
    """Require equivalent executable task resources in all three harness images."""
    if not config.is_corrected_phase1:
        return {"checked": False}
    import docker

    manifest = _runtime_manifest(config)
    images = {
        "react": config.sandbox["image"],
        "nanobot": agents["nanobot"]["image"],
        "openclaw": agents["openclaw"]["image"],
    }
    probe = (
        "import csv,io,json,platform,pandas,pytest; "
        "rows=list(csv.DictReader(io.StringIO('id,value\\na,2\\nb,3\\n'))); "
        "assert sum(int(r['value']) for r in rows)==5; "
        "assert pandas.DataFrame(rows)['value'].astype(int).sum()==5; "
        "print(json.dumps({'python':platform.python_version(),"
        "'pandas':pandas.__version__,'pytest':pytest.__version__}))"
    )
    client = docker.from_env()
    observed: dict[str, Any] = {}
    try:
        for name, image_name in images.items():
            image = client.images.get(image_name)
            expected = str(((manifest.get("images") or {}).get("benchmark" if name == "react" else name) or {}).get("image_id", ""))
            if not expected or image.attrs.get("Id") != expected:
                raise PreflightError(f"Task-tool image ID drifted: {name}")
            container = client.containers.create(image_name, command=["python", "-c", probe], network_disabled=True)
            try:
                container.start()
                waited = container.wait(timeout=30)
                output = container.logs().decode("utf-8", errors="replace")
                if int(waited.get("StatusCode", 1)) != 0:
                    raise PreflightError(f"Task-tool parity probe failed in {name}: {output[-300:]}")
                observed[name] = json.loads(output.strip().splitlines()[-1])
            finally:
                container.remove(force=True)
        if len({json.dumps(value, sort_keys=True) for value in observed.values()}) != 1:
            raise PreflightError(f"Task-tool versions differ across harnesses: {observed}")
        return {"checked": True, "versions": observed["react"]}
    except docker.errors.DockerException as exc:
        raise PreflightError(f"Task-tool parity Docker check failed: {exc}") from exc
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
    'tools': [{
        'type': 'function', 'function': {
            'name': 'add', 'description': 'Add two integers.',
            'parameters': {
                'type': 'object',
                'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                'required': ['a', 'b'], 'additionalProperties': False,
            },
        },
    }],
    'tool_choice': 'auto', 'temperature': 0, 'top_p': 1, 'max_tokens': __MAX_TOKENS__, 'stream': False,
})
calls = ((result.get('choices') or [{}])[0].get('message') or {}).get('tool_calls')
if not calls or calls[0].get('function', {}).get('name') != 'add':
    raise RuntimeError('tool-call-mismatch')
stream_request = urllib.request.Request(
    base + '/chat/completions',
    data=json.dumps({
        'model': os.environ['PHASE1_PROXY_MODEL'],
        'messages': [{'role': 'user', 'content': 'Reply with one short word.'}],
        'temperature': 0, 'top_p': 1, 'max_tokens': __MAX_TOKENS__,
        'stream': True, 'stream_options': {'include_usage': True},
    }).encode(),
    headers=headers, method='POST',
)
with urllib.request.urlopen(stream_request, timeout=30) as response:
    stream_body = response.read().decode('utf-8', errors='replace')
stream_usage = None
stream_chunks = 0
for line in stream_body.splitlines():
    if not line.startswith('data:'):
        continue
    item = line[5:].strip()
    if not item or item == '[DONE]':
        continue
    try:
        chunk = json.loads(item)
    except json.JSONDecodeError:
        continue
    stream_chunks += 1
    if isinstance(chunk, dict) and isinstance(chunk.get('usage'), dict):
        stream_usage = chunk['usage']
if (
    stream_chunks < 1
    or not isinstance(stream_usage, dict)
    or not stream_usage.get('total_tokens')
):
    raise RuntimeError('stream-usage-missing')
print('proxy-ok')
"""
        probe_script = probe_script.replace(
            "__MAX_TOKENS__", str(int(config.generation.get("max_tokens", 2048)))
        )
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
        snapshot = sidecar.seal()
        usage = snapshot.get("usage") or {}
        if int(snapshot.get("calls", 0)) != 2:
            raise PreflightError("The internal model proxy did not record both probe calls")
        if not all(isinstance(usage.get(key), int) and usage[key] > 0
                   for key in ("input_tokens", "output_tokens", "total_tokens")):
            raise PreflightError("The internal model proxy did not capture streamed token usage")
        if endpoint.resolved_model in json.dumps(snapshot, ensure_ascii=False):
            raise PreflightError("The private served model identifier appeared in sidecar trace data")
        return {"enabled": True, "tool_call_verified": True, "stream_usage_verified": True,
                "calls": 2, "usage_present": True}
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
    translation_path = config.root / config.experiment.get(
        "translations_manifest", "benchmark/translations/phase1.yaml"
    )
    translations = (yaml.safe_load(translation_path.read_text(encoding="utf-8")) or {}).get("tasks", {})
    checked = 0
    for task in tasks:
        task_root = config.task_root / task.task_id
        marker = task_root / ".source_sha256"
        source = task_root / "source"
        if not marker.is_file() or not source.is_dir():
            raise PreflightError(f"Prepared task is incomplete: {task.task_id}")
        actual = marker.read_text(encoding="utf-8").strip()
        actual_tree = tree_sha256(source)
        if expected.get(task.task_id) and (actual != expected[task.task_id] or actual_tree != actual):
            raise PreflightError(f"Prepared task hash mismatch: {task.task_id}")
        record = translations.get(task.task_id)
        if not isinstance(record, dict) or record.get("source_sha256") != actual_tree:
            raise PreflightError(f"Translation provenance mismatch: {task.task_id}")
        canonical_prompt = (source / "prompt.txt").read_text(encoding="utf-8")
        if task.instruction_for("english") != canonical_prompt:
            raise PreflightError(f"Canonical English prompt mismatch: {task.task_id}")
        for language in ("english", "hindi", "hinglish"):
            prompt = task.instruction_for(language)
            if not isinstance(prompt, str) or not prompt.strip():
                raise PreflightError(f"Empty {language} prompt: {task.task_id}")
            if prompt != (record.get("instructions") or {}).get(language):
                raise PreflightError(f"Translation overlay mismatch: {task.task_id}/{language}")
            if language != "english" and validate_translation(canonical_prompt, prompt):
                raise PreflightError(f"Protected translation entity mismatch: {task.task_id}/{language}")
        if task.task_id == "016-code-repair-pytest":
            import importlib.util

            oracle_path = source / "oracle_grade.py"
            spec = importlib.util.spec_from_file_location("phase1_preflight_016", oracle_path)
            if spec is None or spec.loader is None:
                raise PreflightError("Cannot load pinned code-repair oracle")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            test_file = source / "fixtures" / "in" / "app" / "test_config.py"
            if hashlib.md5(test_file.read_bytes()).hexdigest() != module.EXPECTED_TEST_HASH:
                raise PreflightError("Code-repair pristine test hash fails its oracle")
        if task.task_id == "050-multitable-join-analysis":
            truth = json.loads((source / "ground_truth.json").read_text(encoding="utf-8"))
            for relative, expected_digest in truth.get("fixture_hashes", {}).items():
                fixture = source / "fixtures" / "in" / relative
                if not fixture.is_file() or hashlib.sha256(fixture.read_bytes()).hexdigest() != expected_digest:
                    raise PreflightError(f"Pristine data fixture fails oracle hash: {relative}")
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
    result["fresh_fixtures"] = check_pilot_fixture_parity(config, tasks)
    result["calibration"] = check_calibration(config)
    result["docker"] = check_docker_image(config)
    result.update(check_native_runtimes(config, agents))
    result["task_tools"] = check_shared_task_tools(config, agents)
    result["proxy_image"] = check_proxy_image(config)
    result["proxy_sidecar"] = check_proxy_sidecar(config, agents)
    result["runtime"] = runtime_preflight(config)
    return result


def check_pilot_fixture_parity(config: ExperimentConfig, tasks: list[Any]) -> dict[str, Any]:
    """Recreate post-hook workspaces in all conditions before any cell exists."""
    if not config.is_corrected_phase1:
        return {"checked": False}
    from runner.runner import workspace_sha256

    fixture_hashes: dict[str, str] = {}
    baseline_oracle: dict[str, float] = {}
    for task in tasks:
        if task.upstream is None:
            raise PreflightError(f"Pilot task lacks pinned upstream contract: {task.task_id}")
        source = config.task_root / task.task_id / task.upstream.source_dir
        observed: set[str] = set()
        for agent in config.experiment["agents"]:
            for language in config.experiment["languages"]:
                with WorkspaceSandbox(None, config.sandbox["image"], "local",
                                      fixtures=source / task.upstream.fixtures_dir) as sandbox:
                    state = prepare_runtime(source, sandbox.root, sandbox.workspace, task.upstream.hooks_module)
                    try:
                        env = {key: str(value) for key, value in state.items()
                               if isinstance(value, (str, int, float))}
                        rendered = render_runtime_template(task.instruction_for(language),
                                                           workspace="/workspace", runtime_env=env)
                        if "$WORKSPACE" in rendered or not rendered.strip() or str(sandbox.workspace) in rendered:
                            raise PreflightError(f"Unrendered or empty pilot prompt: {task.task_id}/{language}")
                        observed.add(workspace_sha256(sandbox.workspace))
                        if task.task_id not in baseline_oracle:
                            grade = run_upstream_oracle(source, task.upstream.oracle_module,
                                                        sandbox.workspace, task.upstream.expected_outcome_score)
                            if grade.outcome_score is None:
                                raise PreflightError(f"Pristine pilot oracle failed: {task.task_id}")
                            baseline_oracle[task.task_id] = grade.outcome_score
                    finally:
                        cleanup_runtime(source, sandbox.root, sandbox.workspace,
                                        task.upstream.hooks_module, state)
        if len(observed) != 1:
            raise PreflightError(f"Fresh post-hook fixtures differ across conditions: {task.task_id}")
        fixture_hashes[task.task_id] = observed.pop()
    return {"checked": True, "tasks": len(fixture_hashes),
            "conditions_per_task": len(config.experiment["agents"]) * len(config.experiment["languages"]),
            "post_hook_hashes": fixture_hashes, "pristine_oracle_scores": baseline_oracle}


def check_calibration(config: ExperimentConfig) -> dict[str, Any]:
    """A corrected Phase I experiment needs a calibrated generation cap."""
    if not config.is_corrected_phase1:
        return {"checked": False}
    from scripts.calibrate_pilot_budget import PROBES

    path = config.root / config.inference.get("calibration_manifest", "")
    if not path.is_file():
        raise PreflightError("The synthetic generation-budget calibration is missing")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError("The calibration manifest is invalid") from exc
    if record.get("experiment_id") != config.experiment_id:
        raise PreflightError("Calibration belongs to another experiment")
    selected = record.get("selected_max_tokens")
    if selected not in (4096, 8192) or selected != config.generation.get("max_tokens"):
        raise PreflightError("The configured generation cap differs from calibration")
    digest = hashlib.sha256(json.dumps(PROBES).encode()).hexdigest()
    if record.get("probes_sha256") != digest:
        raise PreflightError("Synthetic calibration prompts changed")
    passed = {trial.get("probe") for trial in record.get("trials", [])
              if trial.get("cap") == selected and trial.get("valid_tool_call") is True}
    if passed != set(range(len(PROBES))):
        raise PreflightError("The selected generation cap did not pass every calibration probe")
    model_manifest = config.root / config.inference["manifest"]
    if not model_manifest.is_file():
        raise PreflightError("The frozen model manifest is missing")
    model = json.loads(model_manifest.read_text(encoding="utf-8"))
    served = model.get("resolved_model")
    if not isinstance(served, str) or record.get("served_model_sha256") != hashlib.sha256(served.encode()).hexdigest():
        raise PreflightError("The calibrated served model differs from the frozen model")
    return {"checked": True, "selected_max_tokens": selected}


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
