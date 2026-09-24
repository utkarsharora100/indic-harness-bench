import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from benchmark.loader import select_tasks
from runner.config import ExperimentConfig
from runner.hybrid_outcome import (
    HybridOutcomeError,
    _packet,
    _reference,
    _semantic_call,
    _validate_response,
    primary_outcome_score,
    open_store,
)


def test_task001_reference_is_derived_from_pinned_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    fixture = source / "fixtures" / "in" / "input.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"one\ntwo\n")
    ref = _reference("001-file", source)
    assert ref["kind"] == "derived_reference"
    assert ref["expected_line_count"] == 2


def test_packet_separates_reference_from_actual_submission(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "prompt.txt").write_text("Write a report.", encoding="utf-8")
    (source / "ground_truth.json").write_text('{"answer":"gold"}', encoding="utf-8")
    fixture = source / "fixtures" / "in" / "input.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"reference input")
    packet = _packet(
        "004-meeting-summary",
        source,
        {"out/summary.md": b"submitted", "in/input.txt": b"mutated input"},
        None,
    )
    assert packet["reference_answer_not_agent_work"]["facts"] == {"answer": "gold"}
    assert packet["submitted_workspace"][0]["content"] == "submitted"
    assert "gold" not in packet["submitted_workspace"][0]["content"]
    assert packet["fixture_integrity"][0]["path"] == "in/input.txt"
    assert packet["fixture_integrity"][0]["status"] == "modified"


def test_missing_submitted_deliverable_is_not_inferred_from_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "prompt.txt").write_text("Create a summary.", encoding="utf-8")
    (source / "ground_truth.json").write_text('{"summary":"correct"}', encoding="utf-8")
    (source / "fixtures").mkdir()
    packet = _packet("004-meeting-summary", source, {}, None)
    assert packet["reference_answer_not_agent_work"]["facts"]["summary"] == "correct"
    assert packet["submitted_workspace"] == []
    with pytest.raises(HybridOutcomeError, match="missing deliverable"):
        _validate_response(
            json.dumps({"level": 4, "reason": "answer key exists", "submitted_evidence_paths": []}),
            set(),
        )


def test_semantic_judge_cannot_cite_reference_or_unknown_files() -> None:
    with pytest.raises(HybridOutcomeError, match="unknown submitted artifact"):
        _validate_response(
            json.dumps(
                {"level": 4, "reason": "supported", "submitted_evidence_paths": ["reference.json"]}
            ),
            {"out/answer.md"},
        )


@pytest.mark.parametrize(("level", "expected"), [(4, 1.0), (2, 0.5), (0, 0.0)])
def test_primary_score_is_llm_only(level: int, expected: float) -> None:
    assert primary_outcome_score(level) == pytest.approx(expected)


def test_llm_only_score_rejects_out_of_range_values() -> None:
    with pytest.raises(HybridOutcomeError):
        primary_outcome_score(5)


def test_empty_workspace_gets_zero_even_if_oracle_rewards_unchanged_fixtures() -> None:
    assert primary_outcome_score(4, has_submission=False) == 0.0


def test_frozen_main24_configuration_is_the_exact_216_cell_matrix() -> None:
    config = ExperimentConfig.load(Path("configs/phase1.corrected.main24-v1.yaml"))
    selection = yaml.safe_load(
        (config.root / config.experiment["dataset_manifest"]).read_text(encoding="utf-8")
    )
    manifest_ids = [item["task_id"] for item in selection["tasks"]]
    assert config.configured_task_ids == manifest_ids
    assert len(select_tasks(config.task_root, manifest_ids)) == 24
    assert config.experiment["languages"] == ["english", "hindi", "hinglish"]
    assert config.experiment["agents"] == ["react", "nanobot", "openclaw"]
    assert config.experiment["repetitions"] == 1
    assert 24 * 3 * 3 * 1 == 216
    rubric = yaml.safe_load(
        (config.root / config.inference["outcome_rubric"]).read_text(encoding="utf-8")
    )
    assert set(rubric["tasks"]) == set(manifest_ids)


def test_call_history_is_append_only(tmp_path: Path) -> None:
    store = open_store(tmp_path / "calls.sqlite", {"version": "test"})
    try:
        store.execute(
            "INSERT INTO call_attempt VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("s", 1, "error", "request", None, None, None, None, "failure", "now"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.execute("UPDATE call_attempt SET status='completed'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.execute("DELETE FROM call_attempt")
    finally:
        store.close()


def test_semantic_transport_records_usage_and_proxy_events(tmp_path: Path) -> None:
    class Proxy:
        active_cell_id = None

        def begin_cell(self, subject: str) -> None:
            assert self.active_cell_id is None
            self.active_cell_id = subject

        def end_cell(self) -> list[dict]:
            assert self.active_cell_id is not None
            self.active_cell_id = None
            return [{"event_type": "proxy_response", "data": {"usage": {"total_tokens": 13}}}]

    class Client:
        class Completions:
            def create(self, **_kwargs):
                message = type(
                    "Message",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "level": 3,
                                "reason": "The submitted result matches the reference.",
                                "submitted_evidence_paths": ["out/result.md"],
                            }
                        )
                    },
                )()
                return type(
                    "Result",
                    (),
                    {
                        "choices": [type("Choice", (), {"message": message})()],
                        "usage": type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 3})(),
                    },
                )()

        chat = type("Chat", (), {"completions": Completions()})()

    runtime = type(
        "Runtime",
        (),
        {
            "proxy": Proxy(),
            "judge": type("Judge", (), {"client": Client()})(),
            "model_config": {"model": "public-alias"},
        },
    )()
    store = open_store(tmp_path / "calls.sqlite", {"version": "test"})
    try:
        level, reason, usage = _semantic_call(
            subject_id="cell",
            prompt="test",
            submitted_paths={"out/result.md"},
            runtime=runtime,
            store=store,
        )
        assert (level, reason) == (3, "The submitted result matches the reference.")
        assert usage == {"input_tokens": 10, "output_tokens": 3}
        assert store.execute("SELECT count(*) FROM call_attempt").fetchone()[0] == 1
        assert store.execute("SELECT count(*) FROM proxy_event").fetchone()[0] == 1
    finally:
        store.close()
