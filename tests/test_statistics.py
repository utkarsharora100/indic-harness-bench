from analysis.statistics import proportion_interval


def test_proportion_interval_contains_point_estimate():
    lower, upper = proportion_interval(8, 10)
    assert 0.8 >= lower
    assert 0.8 <= upper
