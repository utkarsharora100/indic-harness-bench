import pytest

from analysis.failure_analysis import paired_outcome, validate_category


def test_paired_outcomes():
    assert paired_outcome(True, False, True) == "hindi_only_failure"
    assert paired_outcome(True, True, False) == "hinglish_only_failure"
    assert paired_outcome(True, True, True) == "all_pass"


def test_failure_category_validation():
    assert validate_category("planning") == "planning"
    with pytest.raises(ValueError):
        validate_category("unknown")
