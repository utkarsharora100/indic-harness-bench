from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any

import pytest

from runner.outcome_judge import OutcomeJudge, OutcomeJudgeError, validate_judgment
from runner.outcome_v2 import (
    MAX_CALL_ATTEMPTS,
    _adjudicate_passes,
    _begin_store_session,
    _next_attempt_number,
    _pass_requires_third,
    _perform_judge_call,
    _record_pass,
    _store_proxy_session,
    init_v2_store,
    validate_v14_generation_settings,
)
from runner.proxy import InferenceProxy, ModelProxyError

ROOT = Path(__file__).resolve().parents[1]


def _criterion() -> list[dict[str, Any]]:
    return [{"id": "correct_line_count", "weight": 1.0, "instruction": "Check the exact count."}]


def _packet() -> dict[str, Any]:
    return {
        "task_id": "001-file",
        "reference_answer_not_submitted_work": {"expected_line_count": 4},
        "criterion_contract": {
            "correct_line_count": {
                "submitted_evidence": [{"path": "out/linecount.txt", "present": True}],
                "forced_level": None,
                "max_level": 4,
            }
        },
        "final_workspace_files": [
            {
                "path": "out/linecount.txt",
                "kind": "text",
                "content": "4\n",
            }
        ],
    }


def _judgment(level: int, rationale: str = "The output contains the cited count.") -> str:
    return json.dumps(
        {
            "ratings": {
                "correct_line_count": {
                    "level": level,
                    "rationale": rationale,
                }
            }
        }
    )


@contextmanager
def _fake_upstream(responses: list[tuple[int, str | dict[str, Any]]]):
    state = {"calls": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request_payload = json.loads(self.rfile.read(length))
            state["requests"].append(request_payload)
            index = state["calls"]
            state["calls"] += 1
            status, content = responses[min(index, len(responses) - 1)]
            if isinstance(content, str):
                payload = {
                    "id": f"fake-{index}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "private/model/path",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                }
            else:
                payload = content
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_proxy_call(tmp_path: Path, responses, monkeypatch):
    monkeypatch.setattr("runner.outcome_v2.time.sleep", lambda _seconds: None)
    with _fake_upstream(responses) as (url, state):
        proxy = InferenceProxy(
            url,
            "fake-upstream-key",
            "private/model/path",
            public_model="public-alias",
            max_calls_per_cell=40,
            temperature=0,
            top_p=1,
            max_tokens=8192,
            bind_host="127.0.0.1",
        ).start()
        config = {"model": "public-alias", "base_url": proxy.base_url, "api_key": proxy.client_key}
        judge = OutcomeJudge(config, proxy=proxy, max_tokens=8192, timeout=3)
        runtime = SimpleNamespace(
            judge=judge,
            model_config={"model": "public-alias"},
            secrets=(url, "fake-upstream-key", "private/model/path", proxy.client_key),
            proxy=proxy,
        )
        store = init_v2_store(tmp_path / "integration.sqlite", {"test": "fake-upstream"})
        session_id = "integration-session"
        _begin_store_session(store, "subject", "round", session_id)
        proxy.begin_cell(session_id)
        holder: dict[str, Any] = {"snapshot": {}, "events": []}
        output = None
        error = None
        try:
            output = _perform_judge_call(
                runtime,
                store,
                subject_id="subject",
                round_id="round",
                pass_no=1,
                packet=_packet(),
                criteria=_criterion(),
                reverse=False,
            )
            _record_pass(store, "subject", "round", 1, output[1], output[0], output[3], output[2])
        except Exception as exc:
            error = exc
        finally:
            holder["snapshot"] = proxy.snapshot_cell()
            holder["events"] = proxy.end_cell()
            _store_proxy_session(
                store,
                "subject",
                "round",
                session_id,
                holder["snapshot"],
                holder["events"],
                runtime.secrets,
            )
        judge.close()
        proxy.close()
        return store, state, output, error, holder


def test_proxy_session_cannot_nest_or_discard_events():
    proxy = InferenceProxy("http://127.0.0.1:1/v1", "k", "private/model", bind_host="127.0.0.1")
    proxy.begin_cell("first")
    with pytest.raises(ModelProxyError, match="still active"):
        proxy.begin_cell("second")
    assert proxy.active_cell_id == "first"
    assert proxy.end_cell() == []
    assert proxy.active_cell_id is None
    with pytest.raises(ModelProxyError, match="no cell is active"):
        proxy.end_cell()


def test_contract_only_pass_needs_no_model_call_and_is_recorded(tmp_path):
    from runner.outcome_v2 import _perform_judge_call

    criteria = [
        {"id": "deliverable_a", "weight": 0.5, "instruction": "Evaluate output A."},
        {"id": "deliverable_b", "weight": 0.5, "instruction": "Evaluate output B."},
    ]
    packet = {
        "task_id": "019-incident-runbook-synthesis",
        "reference_answer_not_submitted_work": {"answer": "expected"},
        "final_workspace_files": [],
        "criterion_contract": {
            criterion["id"]: {
                "submitted_evidence": [{"path": f"out/{criterion['id']}.txt", "present": False}],
                "reference_fields": ["answer"],
                "forced_level": 0,
                "max_level": 4,
            }
            for criterion in criteria
        },
    }
    runtime = SimpleNamespace(judge=None)
    store = init_v2_store(tmp_path / "deterministic.sqlite", {"test": "contract-only"})
    ratings, score, usage, response_hash = _perform_judge_call(
        runtime,
        store,
        subject_id="missing-control",
        round_id="test",
        pass_no=1,
        packet=packet,
        criteria=criteria,
        reverse=False,
    )
    attempt = store.execute("SELECT status,input_tokens FROM call_attempt").fetchone()
    assert score == 0.0
    assert all(item["model_level"] is None for item in ratings.values())
    assert attempt["status"] == "deterministic" and attempt["input_tokens"] is None
    assert usage["deterministic"] is True and response_hash
    store.close()


def test_proxy_enforces_frozen_8192_generation_policy():
    proxy = InferenceProxy(
        "http://127.0.0.1:1/v1", "k", "private/model", max_tokens=8192, bind_host="127.0.0.1"
    )
    assert proxy._validate_payload({"temperature": 0, "top_p": 1, "max_tokens": 8192}) is None
    assert proxy._validate_payload({"max_tokens": 2048}) == "max_tokens_must_match_experiment"
    assert proxy._validate_payload({"temperature": 0.2}) == "temperature_must_match_experiment"
    body = {"stream": True}
    proxy._normalize_payload(body)
    assert body == {
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 8192,
        "stream_options": {"include_usage": True},
    }


def test_generation_settings_must_match_v14():
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192},
        "inference": {"proxy": {"max_calls_per_cell": 40}},
    }
    validate_v14_generation_settings(config)
    config["generation"]["max_tokens"] = 2048
    with pytest.raises(Exception, match="differ from the frozen"):
        validate_v14_generation_settings(config)


def test_minimal_response_requires_frozen_criterion_and_level():
    packet = _packet()
    valid = _judgment(4)
    assert validate_judgment(valid, _criterion(), packet)["correct_line_count"]["level"] == 4
    forged_id = valid.replace("correct_line_count", "wrong_id")
    with pytest.raises(OutcomeJudgeError, match="criterion IDs"):
        validate_judgment(forged_id, _criterion(), packet)
    forged_level = valid.replace('"level": 4', '"level": 5')
    with pytest.raises(OutcomeJudgeError, match="Invalid"):
        validate_judgment(forged_level, _criterion(), packet)


def test_pass_adjudication_uses_thresholds_medians_and_review_status():
    criteria = _criterion()
    first = {"score": 0.75, "ratings": {"correct_line_count": {"level": 3}}}
    second = {"score": 1.0, "ratings": {"correct_line_count": {"level": 4}}}
    assert _pass_requires_third(first, second)
    close_passes = {
        1: {
            **first,
            "ratings": {"correct_line_count": {"level": 3, "evidence": [], "rationale": "ok"}},
        },
        2: {
            "score": 0.75,
            "ratings": {"correct_line_count": {"level": 3, "evidence": [], "rationale": "ok"}},
        },
    }
    status, score, ratings, spread, scores = _adjudicate_passes(close_passes, criteria)
    assert (status, score, ratings["correct_line_count"]["level"], spread, scores) == (
        "completed",
        0.75,
        3.0,
        0.0,
        [0.75, 0.75],
    )
    wide = {
        index: {
            "score": level / 4,
            "ratings": {"correct_line_count": {"level": level, "evidence": [], "rationale": "ok"}},
        }
        for index, level in enumerate((0, 2, 4), start=1)
    }
    status, score, ratings, spread, scores = _adjudicate_passes(wide, criteria)
    assert status == "needs_review"
    assert score is None
    assert ratings["correct_line_count"]["level"] == 2
    assert spread == 1.0
    assert scores == [0.0, 0.5, 1.0]


def test_call_and_proxy_event_tables_are_append_only(tmp_path: Path):
    connection = init_v2_store(tmp_path / "append-only.sqlite", {"round": "v3"})
    connection.execute(
        "INSERT INTO call_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("a", "s", "r", 1, 1, "error", "q", None, None, None, None, None, "e", "t0", "t1"),
    )
    connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("UPDATE call_attempt SET status='completed' WHERE attempt_id='a'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM call_attempt WHERE attempt_id='a'")
    connection.close()


def test_call_attempt_numbers_resume_without_reusing_interrupted_attempts(tmp_path: Path):
    connection = init_v2_store(tmp_path / "resume.sqlite", {"round": "v3"})
    for attempt in (1, 2):
        connection.execute(
            "INSERT INTO call_event(attempt_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
            (
                f"a{attempt}",
                "started",
                "now",
                json.dumps(
                    {"subject_id": "s", "round_id": "r", "pass_no": 2, "attempt_no": attempt}
                ),
            ),
        )
    connection.commit()
    assert _next_attempt_number(connection, "s", "r", 2) == 3
    assert MAX_CALL_ATTEMPTS == 3
    connection.close()


def test_fake_endpoint_malformed_json_retries_then_records_usage_and_redacts(tmp_path, monkeypatch):
    store, state, result, error, session = _run_proxy_call(
        tmp_path,
        [(200, "not json"), (200, _judgment(4, "model private/model/path"))],
        monkeypatch,
    )
    assert error is None
    assert result[1] == 1.0
    assert result[2]["input_tokens"] == 12
    assert result[2]["output_tokens"] == 8
    assert result[2]["response_model"] == "public-alias"
    assert state["calls"] == 2
    assert (
        "Validation feedback from your previous response"
        in state["requests"][1]["messages"][0]["content"]
    )
    request_hashes = store.execute(
        "SELECT request_sha256 FROM call_attempt ORDER BY attempt_no"
    ).fetchall()
    assert request_hashes[0][0] != request_hashes[1][0]
    attempts = store.execute(
        "SELECT status,raw_response FROM call_attempt ORDER BY attempt_no"
    ).fetchall()
    assert [row["status"] for row in attempts] == ["error", "completed"]
    assert "private/model/path" not in json.dumps([dict(row) for row in attempts])
    assert "fake-upstream-key" not in json.dumps(session)
    events = [row[0] for row in store.execute("SELECT payload_json FROM proxy_event")]
    assert all("private/model/path" not in value for value in events)
    assert any(json.loads(value).get("model") == "public-alias" for value in events)
    assert store.execute("SELECT COUNT(*) FROM pass_result").fetchone()[0] == 1
    store.close()


def test_fake_endpoint_500_retries_but_deterministic_400_does_not(tmp_path, monkeypatch):
    transient_store, transient_state, result, error, _session = _run_proxy_call(
        tmp_path / "transient",
        [(500, {"error": "temporary"}), (200, _judgment(4))],
        monkeypatch,
    )
    assert error is None and result[1] == 1.0
    assert transient_state["calls"] == 2
    assert transient_store.execute("SELECT COUNT(*) FROM call_attempt").fetchone()[0] == 2
    transient_store.close()

    error_store, error_state, result, error, _session = _run_proxy_call(
        tmp_path / "fatal", [(400, {"error": "policy"})], monkeypatch
    )
    assert result is None
    assert error is not None and "exhausted after 1 attempt" in str(error)
    assert error_state["calls"] == 1
    assert error_store.execute("SELECT status FROM call_attempt").fetchone()[0] == "error"
    error_store.close()


def test_original_v14_database_hash_is_unchanged_by_fake_inference_tests():
    database = ROOT / "data/phase1/corrected/pilot-v14/runs.sqlite"
    before = __import__("hashlib").sha256(database.read_bytes()).hexdigest()
    assert before == "d62e4753eb45fec807427e6866d88a34966b78cfa2fc1815689d2e497cccfecb"
    after = __import__("hashlib").sha256(database.read_bytes()).hexdigest()
    assert after == before
