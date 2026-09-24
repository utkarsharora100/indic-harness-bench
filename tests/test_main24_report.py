from __future__ import annotations

from analysis.main24_report import HARNESSES, LANGUAGES, _summary


def test_main24_summary_uses_task_paired_language_differences() -> None:
    rows = []
    for task_index in range(24):
        for harness_index, harness in enumerate(HARNESSES):
            for language in LANGUAGES:
                language_lift = {"english": 0.0, "hindi": 0.1, "hinglish": -0.05}[language]
                score = 0.3 + 0.05 * harness_index + language_lift + task_index / 1000
                rows.append(
                    {
                        "task_id": f"T{task_index:02}",
                        "agent": harness,
                        "language": language,
                        "outcome_score": score,
                        "judgment_status": "completed",
                        "grade_status": "completed",
                        "process_status": "completed",
                        "oracle_score": score - 0.02,
                        "semantic_level": 3,
                        "initial_prompt_tokens": 100,
                        "total_tokens": 200,
                        "agent_time": 4.0,
                        "end_to_end_time": 5.0,
                        "tool_calls": 2,
                        "failed_tool_calls": 0,
                        "process_score": 0.8,
                        "security_score": 1.0,
                        "hybrid_diagnostic": score * 0.8,
                    }
                )
    conditions, contrasts, coverage = _summary(rows)
    assert len(conditions) == 9
    assert len(contrasts) == 6
    hindi = next(
        row for row in contrasts if row["harness"] == "react" and row["contrast"] == "hindi-English"
    )
    hinglish = next(
        row
        for row in contrasts
        if row["harness"] == "openclaw" and row["contrast"] == "hinglish-English"
    )
    assert abs(hindi["mean_paired_difference"] - 0.1) < 1e-12
    assert hindi["n_tasks"] == 24
    assert abs(hindi["bootstrap_95_lower"] - 0.1) < 1e-12
    assert abs(hindi["bootstrap_95_upper"] - 0.1) < 1e-12
    assert abs(hinglish["mean_paired_difference"] + 0.05) < 1e-12
    assert coverage["n_agent_cells"] == 216


def test_main24_summary_keeps_usage_missingness_explicit() -> None:
    rows = []
    for task in range(24):
        for harness in HARNESSES:
            for language in LANGUAGES:
                rows.append(
                    {
                        "task_id": str(task),
                        "agent": harness,
                        "language": language,
                        "outcome_score": 0.5,
                        "judgment_status": "completed",
                        "grade_status": "completed",
                        "process_status": None,
                        "oracle_score": 0.4,
                        "semantic_level": 2,
                        "initial_prompt_tokens": None,
                        "total_tokens": None,
                        "agent_time": None,
                        "end_to_end_time": None,
                        "tool_calls": None,
                        "failed_tool_calls": None,
                        "process_score": None,
                        "security_score": None,
                        "hybrid_diagnostic": None,
                    }
                )
    _, _, coverage = _summary(rows)
    assert coverage["n_missing_usage"] == 216
    assert coverage["n_missing_process"] == 216
    assert coverage["n_missing_security"] == 216
