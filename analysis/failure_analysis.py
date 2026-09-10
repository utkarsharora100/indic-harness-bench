from __future__ import annotations

FAILURE_CATEGORIES = {
    "instruction_understanding",
    "planning",
    "tool_selection",
    "tool_argument",
    "execution",
    "context_state",
    "error_recovery",
    "completion",
    "evaluation_infrastructure",
}


def validate_category(category: str) -> str:
    if category not in FAILURE_CATEGORIES:
        raise ValueError(f"Unknown failure category: {category}")
    return category


def paired_outcome(english_success: bool, hindi_success: bool, hinglish_success: bool) -> str:
    values = (english_success, hindi_success, hinglish_success)
    if values == (True, True, True):
        return "all_pass"
    if values == (True, False, True):
        return "hindi_only_failure"
    if values == (True, True, False):
        return "hinglish_only_failure"
    if values == (True, False, False):
        return "both_indic_fail"
    if values == (False, True, True):
        return "english_only_failure"
    return "mixed"
