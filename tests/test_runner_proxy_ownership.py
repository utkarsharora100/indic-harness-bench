from __future__ import annotations

import copy
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import runner.runner as runner_module
from runner.config import ExperimentConfig
from runner.inference import InferenceTransientError
from runner.openclaw_language_study import _record_preflight_failure
from runner.runner import ExperimentRunner, plan_experiment_cells


class FakeEndpoint:
    def __init__(self) -> None:
        self.completions = 0
        self.model_checks = 0
        self.status_code = 200

        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                return

            def _reply(self, value: dict) -> None:
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                endpoint.model_checks += 1
                if endpoint.status_code != 200:
                    body = b"private upstream detail"
                    self.send_response(endpoint.status_code)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._reply({"object": "list", "data": [{"id": "served-model"}]})

            def do_POST(self) -> None:  # noqa: N802
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert payload["model"] == "served-model"
                endpoint.completions += 1
                self._reply({
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _config(root: Path) -> ExperimentConfig:
    path = root / "configs" / "phase1.yaml"
    return ExperimentConfig(path, {
        "experiment": {
            "id": "proxy-ownership-test",
            "seed": 17,
            "repetitions": 1,
            "languages": ["english"],
            "agents": ["openclaw"],
            "models": ["university_gpu"],
            "tasks": ["fake-task"],
            "dataset_manifest": "missing.yaml",
        },
        "generation": {"temperature": 0, "top_p": 1, "max_tokens": 64},
        "sandbox": {"mode": "docker", "image": "unused"},
        "inference": {
            "provider": "university_gpu",
            "proxy": {"enabled": True, "public_model": "public-alias", "max_calls_per_cell": 4},
        },
        "storage": {
            "experiment_dir": "data",
            "database": "data/runs.sqlite",
            "traces_dir": "data/traces",
            "results_dir": "data/results",
        },
    })


def test_matrix_planning_does_not_start_proxy_or_open_run_store(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path / "project")
    task = type("Task", (), {"task_id": "fake-task"})()

    def unexpected(*_args, **_kwargs):
        pytest.fail("planning must not create a proxy or run database")

    monkeypatch.setattr(runner_module, "InferenceProxy", unexpected)
    monkeypatch.setattr(runner_module, "RunStore", unexpected)
    cells = plan_experiment_cells(
        config,
        {"openclaw": {"type": "openclaw"}},
        {"university_gpu": {"provider": "university_gpu", "model": "served-model"}},
        [task],
    )
    assert len(cells) == 1
    assert cells[0].cell_id
    assert not (config.root / "data" / "runs.sqlite").exists()


def test_successive_runners_keep_caller_endpoint_and_forward_once(tmp_path) -> None:
    endpoint = FakeEndpoint()
    raw_models = {
        "university_gpu": {
            "provider": "university_gpu",
            "base_url": endpoint.url,
            "api_key": "fake-endpoint-key",
            "model": "served-model",
        }
    }
    original_models = copy.deepcopy(raw_models)
    config = _config(tmp_path / "project")
    first = second = None
    try:
        first = ExperimentRunner(config, {"openclaw": {"type": "openclaw"}}, raw_models)
        first_proxy_url = first.models["university_gpu"]["base_url"]
        assert raw_models == original_models
        task = type("Task", (), {"task_id": "fake-task"})()
        planned = plan_experiment_cells(config, {"openclaw": {}}, raw_models, [task])
        assert first.plan_cells([task])[0].cell_id == planned[0].cell_id
        first.close()
        first = None

        second = ExperimentRunner(config, {"openclaw": {"type": "openclaw"}}, raw_models)
        proxy = second.proxies["university_gpu"]
        assert proxy.upstream_base_url == endpoint.url
        assert second.models["university_gpu"]["base_url"] != first_proxy_url
        assert raw_models == original_models

        proxy.verify_upstream_identity(timeout_seconds=2)
        proxy.begin_cell("ownership-regression")
        payload = json.dumps({
            "model": "public-alias",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 64,
            "stream": False,
        }).encode()
        request = urllib.request.Request(
            f"{proxy.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {proxy.client_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
        proxy.end_cell()
        assert result["choices"][0]["message"]["content"] == "ok"
        assert endpoint.model_checks == 1
        assert endpoint.completions == 1
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        endpoint.close()


def test_endpoint_failures_are_classified_without_body_or_private_url(
    tmp_path, monkeypatch
) -> None:
    import errno
    import urllib.error

    from runner.inference import InferenceConfigurationError, _json_request, list_models

    endpoint = FakeEndpoint()
    try:
        endpoint.status_code = 403
        with pytest.raises(InferenceConfigurationError) as rejected:
            list_models(endpoint.url, "fake-key", timeout_seconds=2)
        assert rejected.value.failure_kind == "model_or_policy_rejection"
        assert rejected.value.status_code == 403
        assert "private upstream detail" not in str(rejected.value)

        endpoint.close()
        endpoint = None
        dead_url = "http://private-university.invalid/v1"

        def connection_refused(*_args, **_kwargs):
            raise urllib.error.URLError(
                ConnectionRefusedError(errno.ECONNREFUSED, "private transport detail")
            )

        monkeypatch.setattr("runner.inference.urllib.request.urlopen", connection_refused)
        with pytest.raises(InferenceTransientError) as refused:
            _json_request(f"{dead_url}/models", api_key="fake-key", timeout_seconds=1)
        assert refused.value.failure_kind == "connection_refused"
        assert dead_url not in str(refused.value)
    finally:
        if endpoint is not None:
            endpoint.close()


def test_health_journal_distinguishes_stale_local_proxy_target(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path / "project")
    config.inference.update({
        "provider": "university_gpu",
        "health_check_interval_seconds": 30,
    })
    config.storage.update({
        "heartbeat": "data/heartbeat.json",
        "journal": "data/journal.jsonl",
    })
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.proxies = {"university_gpu": SimpleNamespace(
        server=object(),
        thread=SimpleNamespace(is_alive=lambda: True),
        upstream_base_url="http://127.0.0.1:9/v1",
        verify_upstream_identity=lambda _timeout: (_ for _ in ()).throw(
            InferenceTransientError("safe error", failure_kind="connection_refused")
        ),
    )}

    class StopAfterJournal(Exception):
        pass

    def stop(_seconds):
        raise StopAfterJournal

    monkeypatch.setattr(runner_module, "sleep", stop)
    with pytest.raises(StopAfterJournal):
        runner._wait_for_pinned_model("cell")
    journal = [
        json.loads(line)
        for line in (config.root / "data/journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert journal[-1]["failure_kind"] == "stale_local_proxy_target"
    assert journal[-1]["upstream_route"] == "local_proxy"


def test_preflight_diagnostic_redacts_private_model_identity(tmp_path) -> None:
    config = _config(tmp_path / "project")
    private_identity = "/private/server/cache/model.gguf"
    config.inference.update({
        "manifest": "data/runtime-model.json",
        "pinned_model_manifest": "data/pinned-model.json",
    })
    for relative in ("data/runtime-model.json", "data/pinned-model.json"):
        path = config.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"resolved_model": private_identity}), encoding="utf-8")

    _record_preflight_failure(config, RuntimeError(f"model failure: {private_identity}"))
    diagnostic = (config.root / "data" / "preflight-failure.json").read_text(encoding="utf-8")
    assert private_identity not in diagnostic
    assert "[REDACTED]" in diagnostic
