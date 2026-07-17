import numpy as np
import pytest
from numba import njit

from numba_utils.arrays import (
    bincount,
    binary_search,
    cumulative_sum,
    fast_clip,
    histogram,
    lower_bound,
    normalize,
    rolling_mean,
    rolling_sum,
    unique_sorted,
    upper_bound,
)

RNG = np.random.default_rng(42)


class TestSearch:
    def test_lower_bound_matches_searchsorted(self):
        arr = np.sort(RNG.integers(0, 100, 500)).astype(np.float64)
        for value in [-1.0, 0.0, 50.0, 99.0, 200.0]:
            assert lower_bound(arr, value) == np.searchsorted(arr, value, "left")

    def test_upper_bound_matches_searchsorted(self):
        arr = np.sort(RNG.integers(0, 100, 500)).astype(np.float64)
        for value in [-1.0, 0.0, 50.0, 99.0, 200.0]:
            assert upper_bound(arr, value) == np.searchsorted(arr, value, "right")

    def test_binary_search_found_and_missing(self):
        arr = np.array([1, 3, 3, 3, 7, 9], dtype=np.int64)
        assert binary_search(arr, 7) == 4
        assert binary_search(arr, 3) == 1  # first occurrence
        assert binary_search(arr, 4) == -1
        assert binary_search(arr, 0) == -1
        assert binary_search(arr, 10) == -1

    def test_empty_array(self):
        arr = np.empty(0, dtype=np.float64)
        assert lower_bound(arr, 1.0) == 0
        assert upper_bound(arr, 1.0) == 0
        assert binary_search(arr, 1.0) == -1

    def test_callable_from_jitted_code(self):
        @njit
        def count_in_range(sorted_arr, lo, hi):
            return upper_bound(sorted_arr, hi) - lower_bound(sorted_arr, lo)

        arr = np.arange(100, dtype=np.float64)
        assert count_in_range(arr, 10.0, 19.0) == 10


class TestFastClip:
    def test_matches_numpy(self):
        arr = RNG.normal(0, 3, 1000)
        np.testing.assert_array_equal(fast_clip(arr, -1.0, 1.0), np.clip(arr, -1.0, 1.0))

    def test_out_buffer_reused(self):
        arr = RNG.normal(0, 3, 100)
        out = np.empty_like(arr)
        result = fast_clip(arr, -1.0, 1.0, out)
        assert result is out

    def test_input_not_mutated(self):
        arr = RNG.normal(0, 3, 100)
        original = arr.copy()
        fast_clip(arr, -1.0, 1.0)
        np.testing.assert_array_equal(arr, original)

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            fast_clip(np.ones(3), 2.0, 1.0)

    def test_wrong_out_length_raises(self):
        with pytest.raises(ValueError):
            fast_clip(np.ones(3), 0.0, 1.0, np.empty(5))


class TestNormalize:
    def test_range_is_zero_one(self):
        arr = RNG.normal(10, 5, 1000)
        result = normalize(arr)
        assert result.min() == 0.0
        assert result.max() == 1.0
        expected = (arr - arr.min()) / (arr.max() - arr.min())
        np.testing.assert_allclose(result, expected)

    def test_constant_array_maps_to_zeros(self):
        np.testing.assert_array_equal(normalize(np.full(10, 7.0)), np.zeros(10))

    def test_int_input_gives_float_output(self):
        result = normalize(np.array([0, 5, 10], dtype=np.int64))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize(np.empty(0))


class TestCumulativeSum:
    def test_matches_numpy(self):
        arr = RNG.normal(0, 1, 1000)
        np.testing.assert_allclose(cumulative_sum(arr), np.cumsum(arr))

    def test_int_dtype_preserved(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = cumulative_sum(arr)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [1, 3, 6])

    def test_empty(self):
        assert cumulative_sum(np.empty(0)).shape == (0,)


class TestRolling:
    def test_rolling_sum_matches_convolve(self):
        arr = RNG.normal(0, 1, 500)
        expected = np.convolve(arr, np.ones(7), mode="valid")
        np.testing.assert_allclose(rolling_sum(arr, 7), expected, atol=1e-9)

    def test_rolling_mean_matches(self):
        arr = RNG.normal(0, 1, 500)
        expected = np.convolve(arr, np.ones(7) / 7, mode="valid")
        np.testing.assert_allclose(rolling_mean(arr, 7), expected, atol=1e-9)

    def test_window_edges(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(rolling_sum(arr, 1), arr)
        np.testing.assert_allclose(rolling_sum(arr, 3), [6.0])

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            rolling_sum(np.ones(3), 0)
        with pytest.raises(ValueError):
            rolling_sum(np.ones(3), 4)


class TestHistograms:
    def test_bincount_matches_numpy(self):
        arr = RNG.integers(0, 50, 10_000)
        np.testing.assert_array_equal(bincount(arr), np.bincount(arr))

    def test_bincount_minlength(self):
        result = bincount(np.array([0, 1], dtype=np.int64), 10)
        assert result.shape == (10,)

    def test_bincount_negative_raises(self):
        with pytest.raises(ValueError):
            bincount(np.array([1, -1], dtype=np.int64))

    def test_histogram_matches_numpy(self):
        arr = RNG.normal(0, 1, 10_000)
        expected, _ = np.histogram(arr, bins=20, range=(-3.0, 3.0))
        np.testing.assert_array_equal(histogram(arr, 20, -3.0, 3.0), expected)

    def test_histogram_int_input(self):
        arr = RNG.integers(0, 100, 10_000)
        expected, _ = np.histogram(arr, bins=10, range=(0, 100))
        np.testing.assert_array_equal(histogram(arr, 10, 0, 100), expected)

    def test_histogram_invalid_args_raise(self):
        with pytest.raises(ValueError):
            histogram(np.ones(3), 0, 0.0, 1.0)
        with pytest.raises(ValueError):
            histogram(np.ones(3), 5, 1.0, 1.0)


class TestUniqueSorted:
    def test_matches_numpy_unique(self):
        arr = np.sort(RNG.integers(0, 100, 5000))
        np.testing.assert_array_equal(unique_sorted(arr), np.unique(arr))

    def test_all_equal(self):
        np.testing.assert_array_equal(unique_sorted(np.full(10, 3.0)), [3.0])

    def test_no_duplicates(self):
        arr = np.arange(10, dtype=np.float64)
        np.testing.assert_array_equal(unique_sorted(arr), arr)

    def test_empty(self):
        assert unique_sorted(np.empty(0)).shape == (0,)

    def test_dtype_preserved(self):
        assert unique_sorted(np.array([1, 1, 2], dtype=np.int32)).dtype == np.int32
