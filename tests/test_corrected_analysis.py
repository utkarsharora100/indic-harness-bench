from analysis.corrected import bootstrap_paired_deltas, paired_task_counts


def test_language_contrasts_keep_every_available_pair_when_another_language_is_missing():
    rows = []
    for task, english, hindi, hinglish in (
        ("a", 0.2, 0.3, 0.4),
        ("b", 0.4, 0.5, None),
        ("c", 0.6, 0.5, 0.7),
    ):
        for language, score in (
            ("english", english),
            ("hindi", hindi),
            ("hinglish", hinglish),
        ):
            if score is not None:
                rows.append({
                    "task_id": task,
                    "agent": "nanobot",
                    "language": language,
                    "outcome_score": score,
                    "grade_status": "completed",
                })

    assert paired_task_counts(rows) == {"nanobot": {"hindi": 3, "hinglish": 2}}
    intervals = bootstrap_paired_deltas(rows, repetitions=200, seed=11)
    assert intervals["nanobot"]["hindi"]["n_tasks"] == 3
    assert intervals["nanobot"]["hinglish"]["n_tasks"] == 2
    assert intervals == bootstrap_paired_deltas(rows, repetitions=200, seed=11)
