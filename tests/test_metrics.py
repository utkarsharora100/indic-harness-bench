from analysis.metrics import language_deltas, success_rates


def test_language_metrics():
    rows = [
        {"language": "english", "success": 1},
        {"language": "english", "success": 1},
        {"language": "hindi", "success": 1},
        {"language": "hindi", "success": 0},
        {"language": "hinglish", "success": 1},
    ]
    rates = success_rates(rows)
    assert rates["english"] == 1.0
    assert rates["hindi"] == 0.5
    assert rates["hinglish"] == 1.0
    assert language_deltas(rows)["hindi_minus_english"] == -0.5
