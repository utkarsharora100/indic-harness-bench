import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from analysis.corrected import trace_links, task_balanced_lift
from benchmark.canonical import blob_tree_sha256, materialize_blobs, task_blobs
from runner.config import ExperimentConfig
from runner.preflight import PreflightError, check_calibration
from runner.proxy_sidecar import ProxySidecar
from runner.inference import InferenceTransientError
from runner.runner import ExperimentRunner, _first_proxy_prompt_tokens
from runner.log import RunStore
from scripts.calibrate_pilot_budget import PROBES
from scripts.prepare_phase1 import tree_sha256


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)


def test_canonical_source_bytes_survive_windows_checkout_conversion(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    task = repository / "tasks" / "001-file"
    task.mkdir(parents=True)
    (task / "prompt.txt").write_bytes(b"First line\nSecond line\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture")
    task_blobs_at_head = task_blobs(repository, "HEAD", "001-file")
    assert task_blobs_at_head["prompt.txt"][0] == b"First line\nSecond line\n"
    materialized = tmp_path / "prepared"
    materialize_blobs(task_blobs_at_head, materialized)
    assert (materialized / "prompt.txt").read_bytes() == b"First line\nSecond line\n"
    assert tree_sha256(materialized) == blob_tree_sha256(task_blobs_at_head)
    (task / "prompt.txt").write_bytes(b"First line\r\nSecond line\r\n")
    assert task_blobs(repository, "HEAD", "001-file")["prompt.txt"][0] != (task / "prompt.txt").read_bytes()


def _config(tmp_path: Path, version: str = "corrected-v13") -> ExperimentConfig:
    path = tmp_path / "configs" / "pilot.yaml"
    path.parent.mkdir()
    return ExperimentConfig(path, {
        "experiment": {"name": f"pilot-{version}", "version": version},
        "generation": {"max_tokens": 4096},
        "inference": {
            "calibration_manifest": "data/calibration.json",
            "manifest": "data/model.json",
        },
    })


@pytest.mark.parametrize("version", ["corrected-v13", "corrected-v14"])
def test_calibration_freezes_cap_model_and_probes(tmp_path: Path, version: str) -> None:
    config = _config(tmp_path, version)
    data = tmp_path / "data"
    data.mkdir()
    (data / "model.json").write_text(json.dumps({"resolved_model": "private-id"}), encoding="utf-8")
    calibration = {
        "experiment_id": config.experiment_id,
        "selected_max_tokens": 4096,
        "served_model_sha256": hashlib.sha256(b"private-id").hexdigest(),
        "probes_sha256": hashlib.sha256(json.dumps(PROBES).encode()).hexdigest(),
        "trials": [{"cap": 4096, "probe": i, "valid_tool_call": True} for i in range(len(PROBES))],
    }
    path = data / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    assert check_calibration(config)["selected_max_tokens"] == 4096
    config.generation["max_tokens"] = 8192
    with pytest.raises(PreflightError, match="generation cap"):
        check_calibration(config)
    config.generation["max_tokens"] = 4096
    calibration["served_model_sha256"] = hashlib.sha256(b"other-model").hexdigest()
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(PreflightError, match="served model"):
        check_calibration(config)


def test_proxy_seal_requires_complete_observable_calls(tmp_path: Path) -> None:
    sidecar = ProxySidecar(
        image="fake", internal_network="internal", egress_network="bridge",
        upstream_base_url="http://private.invalid", upstream_key="secret",
        resolved_model="private-model", public_model="public",
        trace_root=tmp_path, cell_id="test",
    )
    class Container:
        def kill(self, **kwargs):
            assert kwargs == {"signal": "SIGTERM"}

        def wait(self, **kwargs):
            assert kwargs == {"timeout": 10}

    sidecar.container = Container()
    trace = sidecar.host_trace / "proxy.json"
    trace.write_text(json.dumps({"calls": 2, "events": [
        {"event_type": "proxy_request"}, {"event_type": "proxy_response"},
        {"event_type": "proxy_request"},
    ]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete model proxy trace"):
        sidecar.seal()
    trace.write_text(json.dumps({"calls": 1, "events": [
        {"event_type": "proxy_request"},
        {"event_type": "proxy_error", "data": {"error": "upstream_unreachable"}},
    ]}), encoding="utf-8")
    with pytest.raises(InferenceTransientError):
        sidecar.seal()


def test_trace_index_exposes_workspace_and_stop_reason(tmp_path: Path) -> None:
    row = {
        "run_id": "cell", "cell_id": "cell", "task_id": "001-file", "language": "hindi",
        "agent": "react", "repetition": 0, "trace_path": str(tmp_path / "trace.jsonl"),
        "metadata_json": json.dumps({"workspace_archive": str(tmp_path / "workspace.tar.gz"),
                                     "trace_complete": True, "agent_stop_reason": "max_tokens"}),
    }
    link = trace_links(tmp_path / "runs.sqlite", [row])[0]
    assert link["workspace_archive"].endswith("workspace.tar.gz")
    assert link["trace_complete"] is True
    assert link["agent_stop_reason"] == "max_tokens"


def test_initial_prompt_usage_uses_first_proxy_response_only() -> None:
    events = [
        {"event_type": "proxy_error", "data": {"status": 503}},
        {"event_type": "proxy_response", "data": {"usage": {"prompt_tokens": 123}}},
        {"event_type": "proxy_response", "data": {"usage": {"prompt_tokens": 456}}},
    ]
    assert _first_proxy_prompt_tokens(events) == 123


def test_experiment_resume_rejects_runtime_image_manifest_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.experiment.update({
        "dataset_manifest": "data/dataset.yaml", "translations_manifest": "data/translations.yaml",
        "agents_manifest": "data/agents.yaml",
    })
    config.inference["runtime_manifest"] = "data/runtime.json"
    data = tmp_path / "data"
    data.mkdir()
    for name in ("dataset.yaml", "translations.yaml", "agents.yaml", "model.json",
                 "runtime.json", "calibration.json"):
        (data / name).write_text("original", encoding="utf-8")
    (data / "dataset.yaml").write_text("{}", encoding="utf-8")
    # The fingerprint routine includes code paths under the experiment root.
    for name in ("runner/runner.py", "agents/react.py", "agents/container.py",
                 "runner/proxy.py", "scripts/proxy_sidecar.py", "runner/grader.py",
                 "runner/judge.py", "benchmark/upstream.py", "runner/preflight.py",
                 "scripts/calibrate_pilot_budget.py", "scripts/prepare_phase1.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("code", encoding="utf-8")
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.store = RunStore(data / "runs.sqlite")
    try:
        runner.store.ensure_experiment(config.experiment_id, runner._safe_config(), {}, "now")
        runner.assert_experiment_identity()
        (data / "runtime.json").write_text("changed image ID", encoding="utf-8")
        with pytest.raises(ValueError, match="identity changed"):
            runner.assert_experiment_identity()
    finally:
        runner.store.close()


def test_pristine_baseline_lift_is_task_balanced() -> None:
    rows = [
        {"task_id": "a", "agent": "react", "language": "english", "outcome_score": 0.7, "grade_status": "completed"},
        {"task_id": "a", "agent": "react", "language": "english", "outcome_score": 0.9, "grade_status": "completed"},
        {"task_id": "b", "agent": "react", "language": "english", "outcome_score": 0.3, "grade_status": "completed"},
    ]
    # First average attempts within task: (0.8 - 0.6 + 0.3 - 0.1) / 2.
    assert task_balanced_lift(rows, {"a": 0.6, "b": 0.1}) == {"react:english": pytest.approx(0.2)}


def test_trace_writer_redacts_upstream_and_proxy_credentials(tmp_path: Path) -> None:
    runner = object.__new__(ExperimentRunner)
    config = _config(tmp_path)
    config.data["storage"] = {"traces_dir": "data/traces"}
    runner.config = config
    runner.proxies = {"model": type("Proxy", (), {
        "upstream_key": "private-key", "upstream_base_url": "http://private.invalid",
        "resolved_model": "/server/private-model.gguf",
    })()}
    runner._append_trace("cell", [{"data": {
        "message": "private-key http://private.invalid /server/private-model.gguf proxy-key"
    }}], {"api_key": "proxy-key"})
    content = (tmp_path / "data" / "traces" / "cell.jsonl").read_text(encoding="utf-8")
    for secret in ("private-key", "http://private.invalid", "/server/private-model.gguf", "proxy-key"):
        assert secret not in content
