import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.outcome_report_v8 import _write_review_sample
from runner.outcome_calibration import _base_outputs, _fixtures
from runner.outcome_judge import load_rubric
from runner.outcome_v8_judge import (
    V8JudgeError,
    impossible_credit,
    judge_subject,
    open_store,
    parse_challenges,
    parse_ratings,
    parse_verification,
    register_subject,
    weighted_score,
)
from runner.outcome_v8_packet import _stable_validation, build_comparison_packet

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data/phase1/tasks-v13"
RUBRIC = load_rubric(ROOT / "configs/outcome-rubric.pilot-v14-v8.yaml")


def test_good_control_exact_fields_are_consistent():
    for task_id in RUBRIC["tasks"]:
        source = TASK_ROOT / task_id / "source"
        files = _fixtures(source)
        files.update({key: value.encode() for key, value in _base_outputs(task_id, source).items()})
        validation = {"returncode": 0, "timed_out": False} if task_id.startswith("016") else None
        packet = build_comparison_packet(task_id, source, files, RUBRIC, validation)
        observed = {
            item["id"]: item["observed"] for item in packet["submitted_answer"]["observations"]
        }
        assert all(value is not None for value in observed.values()), task_id
        contradictions = [
            check["id"]
            for check in packet["reference_requirements"]
            if check["kind"] == "exact" and impossible_credit(check, observed[check["id"]], 4)
        ]
        assert contradictions == [], (task_id, contradictions)
        ratings = {check["id"]: {"level": 4} for check in packet["reference_requirements"]}
        assert weighted_score(packet, ratings) == 1.0


def _one_packet(output: bytes | None) -> dict:
    task_id = "001-file"
    source = TASK_ROOT / task_id / "source"
    files = _fixtures(source)
    if output is not None:
        files["out/linecount.txt"] = output
    return build_comparison_packet(task_id, source, files, RUBRIC)


def test_missing_submission_cannot_inherit_reference_credit():
    packet = _one_packet(None)
    count = packet["reference_requirements"][0]
    assert count["id"] == "correct_line_count.value"
    assert impossible_credit(count, None, 4)


def test_exact_wrong_count_cannot_receive_full_credit():
    packet = _one_packet(b"999\n")
    count = packet["reference_requirements"][0]
    assert impossible_credit(count, "999\n", 4)
    assert packet["submitted_answer"]["observations"][0]["exact_match"] is False


def test_unsupported_evidence_id_is_rejected():
    packet = _one_packet(b"4\n")
    check = packet["reference_requirements"][0]
    with pytest.raises(V8JudgeError, match="submitted evidence"):
        parse_ratings(
            '{"ratings":{"correct_line_count.value":'
            '{"level":4,"evidence_id":"REFERENCE","observation":"four"}}}',
            packet,
            [check],
        )


def test_flat_rating_map_is_accepted_without_changing_level():
    packet = _one_packet(b"4\n")
    check = packet["reference_requirements"][0]
    result = parse_ratings(
        '{"correct_line_count.value":'
        '{"level":4,"evidence_id":"A01","observation":"Four lines."}}',
        packet,
        [check],
    )
    assert result["correct_line_count.value"]["level"] == 4


def test_singular_empty_challenge_alias_is_accepted():
    packet = _one_packet(b"4\n")
    assert parse_challenges('{"challenge":[]}', packet["reference_requirements"][:1]) == []


def test_repeated_valid_rating_map_means_no_verifier_challenge():
    packet = _one_packet(b"4\n")
    check = packet["reference_requirements"][0]
    proposed = {check["id"]: {"level": 4, "evidence_id": "A01", "observation": "Four."}}
    response = json.dumps(proposed)
    assert parse_verification(response, packet, [check], proposed) == []


def test_boolean_verification_identifies_unsupported_claim():
    packet = _one_packet(b"4\n")
    check = packet["reference_requirements"][0]
    proposed = {check["id"]: {"level": 4, "evidence_id": "A01"}}
    response = json.dumps(
        {"verification": {check["id"]: {"supported": False, "reason": "Value missing"}}}
    )
    assert parse_verification(response, packet, [check], proposed) == [
        {"id": check["id"], "reason": "Value missing"}
    ]


def test_submitted_observation_id_can_support_credit():
    packet = _one_packet(b"4\n")
    check = packet["reference_requirements"][1]
    response = json.dumps(
        {check["id"]: {"level": 4, "evidence_id": check["id"],
                       "observation": "The input hash is unchanged."}}
    )
    assert parse_ratings(response, packet, [check])[check["id"]]["level"] == 4


def test_pytest_elapsed_time_does_not_change_evidence_hash():
    first = _stable_validation({"returncode": 0, "stdout": "3 passed in 0.06s\n"})
    second = _stable_validation({"returncode": 0, "stdout": "3 passed in 0.19s\n"})
    assert first == second


class FakeRuntime:
    def __init__(self, malformed_first: bool = False):
        self.calls = 0
        self.malformed_first = malformed_first
        self.secrets = ("private-key-test",)
        self.model_config = {"model": "public-model"}
        self.judge = SimpleNamespace(
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            )
        )

    @contextmanager
    def cell_session(self, cell_id):
        yield {"events": [{"type": "done", "model": "public-model",
                           "secret": "private-key-test"}]}

    def create(self, **kwargs):
        self.calls += 1
        if self.malformed_first and self.calls == 1:
            content = "not json"
        else:
            request = json.loads(kwargs["messages"][1]["content"])
            if "verification" in request["return"]:
                content = json.dumps(
                    {"verification": {
                        key: {"supported": True, "reason": "matches"}
                        for key in request["return"]["verification"]
                    }}
                )
            else:
                observations = {
                    item["id"]: item
                    for item in request["comparison"]["agent_answer"]["observations"]
                }
                content = json.dumps(
                    {
                        "ratings": {
                            item["id"]: {
                                "level": 4,
                                "evidence_id": observations[item["id"]]["evidence_id"],
                                "observation": "Submitted value matches the reference.",
                            }
                            for item in request["comparison"]["reference_answer"]
                        }
                    }
                )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def test_fake_endpoint_judgment_persists_retry_and_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("runner.outcome_v8_judge.time.sleep", lambda _: None)
    packet = _one_packet(b"4\n")
    db = open_store(tmp_path / "judge.sqlite", {"version": "v8-test"})
    try:
        register_subject(db, "cell-1", "pilot", "001-file", packet)
        runtime = FakeRuntime(malformed_first=True)
        result = judge_subject(runtime, db, "cell-1", packet)
        assert result["status"] == "completed"
        assert result["score"] == 1
        assert runtime.calls == 5
        assert db.execute("SELECT COUNT(*) FROM call_attempt").fetchone()[0] == 5
        assert db.execute("SELECT SUM(total_tokens) FROM call_attempt").fetchone()[0] == 75
        assert db.execute("SELECT COUNT(*) FROM proxy_event").fetchone()[0] == 1
        assert "private-key-test" not in db.execute(
            "SELECT payload_json FROM proxy_event"
        ).fetchone()[0]
        assert judge_subject(runtime, db, "cell-1", packet)["score"] == 1
        assert runtime.calls == 5
    finally:
        db.close()


def test_v8_human_review_sample_keeps_unscored_cells_and_meets_quotas(tmp_path):
    tasks = sorted(RUBRIC["tasks"])
    rows = []
    packets = {}
    for task_index, task_id in enumerate(tasks):
        for harness in ("react", "nanobot", "openclaw"):
            for language in ("english", "hindi", "hinglish"):
                run_id = f"{task_index}-{harness}-{language}"
                missing = (
                    task_id.startswith("050")
                    and harness == "nanobot"
                    and language == "hinglish"
                )
                needs_review = task_id.startswith("050") or (
                    task_id.startswith("025") and language == "hinglish"
                )
                score = None if needs_review or missing else ((len(rows) % 9) + 1) / 10
                rows.append(
                    {
                        "run_id": run_id,
                        "task_id": task_id,
                        "harness": harness,
                        "language": language,
                        "status": (
                            "missing_agent_artifact" if missing else
                            "needs_review" if needs_review else "completed"
                        ),
                        "outcome_score": score,
                        "oracle_score": 0.5,
                        "workspace_archive": "archive.tar",
                    }
                )
                packets[run_id] = {
                    "packet": {
                        "canonical_question": "question",
                        "reference_requirements": [{"id": "r1"}],
                        "submitted_answer": "answer",
                        "criterion_weights": {"r1": 1},
                    }
                }
    _write_review_sample(tmp_path, rows, packets)
    key = json.loads((tmp_path / "human-review-unblinding-key.json").read_text())
    assert len(key) == 12
    assert Counter(item["language"] for item in key) == {
        "english": 4, "hindi": 4, "hinglish": 4
    }
    assert Counter(item["harness"] for item in key) == {
        "react": 4, "nanobot": 4, "openclaw": 4
    }
    assert all(count >= 2 for count in Counter(item["task_id"] for item in key).values())
    assert {item["task_id"] for item in key} == set(tasks)
    assert any(item["llm_score"] is None for item in key)
    assert {item["disagreement_band"] for item in key} >= {"low", "middle", "high"}
