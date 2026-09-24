from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from analysis.outcome_report_v2 import _write_human_review, select_review_sample
from runner.outcome_judge import load_rubric
from runner.outcome_v2 import HARNESSES, KNOWN_MISSING, LANGUAGES, TASK_IDS


def test_review_sample_meets_all_frozen_strata_and_is_deterministic():
    rows = []
    index = 0
    for task in TASK_IDS:
        for harness in HARNESSES:
            for language in LANGUAGES:
                if (task, language, harness) == KNOWN_MISSING:
                    continue
                rows.append(
                    {
                        "run_id": f"run-{index}",
                        "cell_id": f"cell-{index}",
                        "task_id": task,
                        "harness": harness,
                        "language": language,
                        "status": "completed",
                        "score": index / 44,
                        "oracle_score": 0.5,
                        "pass_scores_json": None,
                    }
                )
                index += 1
    first = select_review_sample(rows, seed=1701)
    second = select_review_sample(rows, seed=1701)
    assert [row["run_id"] for row in first] == [row["run_id"] for row in second]
    assert len(first) == 12
    assert Counter(row["language"] for row in first) == {language: 4 for language in LANGUAGES}
    assert Counter(row["harness"] for row in first) == {harness: 4 for harness in HARNESSES}
    assert all(count >= 2 for count in Counter(row["task_id"] for row in first).values())
    assert {row["disagreement_band"] for row in first} == {"low", "middle", "high"}


def test_blinded_packet_omits_machine_score_and_condition(tmp_path: Path) -> None:
    store = sqlite3.connect(":memory:")
    store.row_factory = sqlite3.Row
    store.execute("CREATE TABLE evidence_packet (run_id TEXT, packet_json TEXT)")
    store.execute(
        "INSERT INTO evidence_packet VALUES (?,?)",
        (
            "run-1",
            json.dumps(
                {
                    "criterion_contract": {
                        "correct_line_count": {
                            "submitted_evidence": [{"path": "out/linecount.txt", "present": True}],
                            "forced_level": 0,
                            "max_level": 4,
                        }
                    }
                }
            ),
        ),
    )
    selected = [
        {
            "run_id": "run-1",
            "cell_id": "cell-1",
            "task_id": "001-file",
            "language": "hindi",
            "harness": "react",
            "score": 0.4,
            "oracle_score": 1.0,
            "disagreement_band": "high",
        }
    ]
    rubric = load_rubric(
        Path(__file__).resolve().parents[1] / "configs/outcome-rubric.pilot-v14-v7.yaml"
    )
    _write_human_review(tmp_path, selected, rubric, store)
    packet = json.loads((tmp_path / "human-review-packet.json").read_text(encoding="utf-8"))[0]
    assert "hindi" not in json.dumps(packet)
    assert "react" not in json.dumps(packet)
    assert "forced_level" not in json.dumps(packet)
    assert "max_level" not in json.dumps(packet)
    assert packet["evidence_packet"]["criterion_contract"]["correct_line_count"][
        "submitted_evidence"
    ]
