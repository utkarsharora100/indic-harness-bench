from analysis.outcome_report import _review_sample, _task_language_deltas


def test_task_level_paired_language_deltas_use_only_complete_scored_cells():
    rows = []
    for language, score in (("english", 0.5), ("hindi", 0.75), ("hinglish", None)):
        rows.append(
            {
                "task_id": "019-incident-runbook-synthesis",
                "agent": "react",
                "language": language,
                "grade_status": "completed" if score is not None else "needs_review",
                "outcome_score": score,
            }
        )
    deltas = _task_language_deltas(rows)
    assert deltas == {"019-incident-runbook-synthesis": {"react": {"hindi": 0.25}}}


def test_blinded_sample_is_balanced_across_languages_and_harnesses(tmp_path):
    rows = []
    for task in ("001-file", "016-code-repair-pytest", "019-incident-runbook-synthesis"):
        for language in ("english", "hindi", "hinglish"):
            for agent in ("react", "nanobot", "openclaw"):
                archive = tmp_path / f"{task}-{language}-{agent}.tar.gz"
                archive.touch()
                rows.append(
                    {
                        "task_id": task,
                        "language": language,
                        "agent": agent,
                        "grade_status": "completed",
                        "archive_path": str(archive),
                    }
                )
    first = _review_sample(rows, 17)
    second = _review_sample(rows, 17)
    assert [row["archive_path"] for row in first] == [row["archive_path"] for row in second]
    assert len(first) == 12
    assert {row["language"] for row in first} == {"english", "hindi", "hinglish"}
    assert {row["agent"] for row in first} == {"react", "nanobot", "openclaw"}
