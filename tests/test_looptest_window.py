from distil._looptest.window import mean_of_tail, rolling_max, tail_window


def test_tail_window_returns_last_n_values():
    assert tail_window([1, 2, 3], 2) == [2, 3]
    assert tail_window([10, 20, 30, 40], 2) == [30, 40]


def test_tail_window_shorter_than_size_returns_all_values():
    assert tail_window([10], 2) == [10]


def test_tail_window_size_zero_or_negative_returns_empty():
    assert tail_window([1, 2, 3], 0) == []
    assert tail_window([1, 2, 3], -1) == []


def test_mean_of_tail_uses_requested_window():
    assert mean_of_tail([1, 2, 3], 2) == 2.5


def test_mean_of_tail_uses_actual_window_length_when_values_shorter():
    assert mean_of_tail([10], 2) == 10.0


def test_mean_of_tail_size_zero_does_not_raise():
    assert mean_of_tail([1, 2, 3], 0) == 0.0


def test_rolling_max_returns_one_max_per_window():
    assert rolling_max([1, 3, 2], 2) == [3, 3]
    assert rolling_max([10, 20], 1) == [10, 20]


def test_rolling_max_size_equal_to_length_returns_full_span_max():
    assert rolling_max([1, 2, 3], 3) == [3]


def test_rolling_max_size_larger_than_length_returns_full_span_max():
    assert rolling_max([1, 2], 5) == [2]


def test_rolling_max_size_zero_or_negative_returns_empty():
    assert rolling_max([1, 2, 3], 0) == []
    assert rolling_max([1, 2, 3], -1) == []


def test_rolling_max_empty_values_returns_empty():
    assert rolling_max([], 2) == []
