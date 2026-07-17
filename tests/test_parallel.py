import numpy as np
import pytest

from numba_utils.parallel import (
    SERIAL_THRESHOLD,
    parallel_histogram,
    parallel_prefix_sum,
    parallel_reduce,
    parallel_sum,
    parallel_topk,
)

RNG = np.random.default_rng(7)
LARGE = SERIAL_THRESHOLD * 4
SMALL = 1000


class TestParallelSum:
    def test_matches_numpy_large_and_small(self):
        for n in (SMALL, LARGE):
            arr = RNG.normal(0.0, 1.0, n)
            assert parallel_sum(arr) == pytest.approx(np.sum(arr), rel=1e-12)

    def test_exact_on_ints(self):
        arr = RNG.integers(0, 100, LARGE).astype(np.int64)
        assert parallel_sum(arr) == float(arr.sum())

    def test_empty(self):
        assert parallel_sum(np.empty(0)) == 0.0


class TestParallelReduce:
    def test_sums_kernel_over_indices(self):
        @parallel_reduce
        def identity(i):
            return float(i)

        for n in (SMALL, LARGE):
            expected = n * (n - 1) / 2
            assert identity(n) == pytest.approx(expected, rel=1e-12)

    def test_zero_n(self):
        @parallel_reduce
        def one(i):
            return 1.0

        assert one(0) == 0.0

    def test_negative_n_raises(self):
        @parallel_reduce
        def one(i):
            return 1.0

        with pytest.raises(ValueError):
            one(-1)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            parallel_reduce(42)


class TestParallelHistogram:
    def test_bit_exact_with_serial_large_and_small(self):
        from numba_utils.arrays import histogram

        for n in (SMALL, LARGE):
            arr = RNG.normal(0.0, 1.0, n)
            np.testing.assert_array_equal(
                parallel_histogram(arr, 32, -4.0, 4.0),
                histogram(arr, 32, -4.0, 4.0),
            )

    def test_matches_numpy(self):
        arr = RNG.normal(0.0, 1.0, LARGE)
        expected, _ = np.histogram(arr, bins=64, range=(-4.0, 4.0))
        np.testing.assert_array_equal(
            parallel_histogram(arr, 64, -4.0, 4.0), expected
        )

    def test_invalid_args_raise(self):
        arr = RNG.normal(0.0, 1.0, LARGE)
        with pytest.raises(ValueError):
            parallel_histogram(arr, 0, 0.0, 1.0)
        with pytest.raises(ValueError):
            parallel_histogram(arr, 8, 1.0, 1.0)


class TestParallelPrefixSum:
    def test_matches_cumsum_large_and_small(self):
        for n in (SMALL, LARGE):
            arr = RNG.normal(0.0, 1.0, n)
            np.testing.assert_allclose(
                parallel_prefix_sum(arr), np.cumsum(arr), rtol=1e-9, atol=1e-9
            )

    def test_exact_on_ints(self):
        arr = RNG.integers(0, 100, LARGE).astype(np.int64)
        np.testing.assert_array_equal(
            parallel_prefix_sum(arr), np.cumsum(arr).astype(np.float64)
        )

    def test_out_buffer_reused(self):
        arr = RNG.normal(0.0, 1.0, SMALL)
        out = np.empty(SMALL)
        assert parallel_prefix_sum(arr, out) is out

    def test_wrong_out_length_raises(self):
        with pytest.raises(ValueError):
            parallel_prefix_sum(np.ones(10), np.empty(5))


class TestParallelTopk:
    def test_matches_serial_large_and_small(self):
        from numba_utils.algorithms import topk

        for n in (SMALL, LARGE):
            arr = RNG.normal(0.0, 1.0, n)
            for k in (1, 10, 100):
                np.testing.assert_array_equal(
                    parallel_topk(arr, k), topk(arr, k)
                )

    def test_duplicates_not_inflated(self):
        arr = np.zeros(LARGE)
        arr[123] = 9.0
        result = parallel_topk(arr, 3)
        np.testing.assert_array_equal(result, [9.0, 0.0, 0.0])

    def test_int_dtype(self):
        arr = RNG.integers(-1000, 1000, LARGE).astype(np.int64)
        expected = np.sort(arr)[-5:][::-1]
        np.testing.assert_array_equal(parallel_topk(arr, 5), expected)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            parallel_topk(np.ones(10), 0)
        with pytest.raises(ValueError):
            parallel_topk(np.ones(10), 11)
