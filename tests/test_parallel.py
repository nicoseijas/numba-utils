import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from numba_utils.parallel import (
    SERIAL_THRESHOLD,
    chunked_reduce,
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

    def test_ignores_nan(self):
        # An unfiltered NaN indexes out of the private counts row
        # (int(NaN) is INT64_MIN) — must be skipped like out-of-range.
        arr = RNG.normal(0.0, 1.0, LARGE)
        arr[::1000] = np.nan
        expected, _ = np.histogram(
            arr[~np.isnan(arr)], bins=32, range=(-4.0, 4.0)
        )
        np.testing.assert_array_equal(
            parallel_histogram(arr, 32, -4.0, 4.0), expected
        )


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

    def test_thread_count_overshooting_chunks(self):
        # Ceil-division chunks overshoot n when threads >= ~sqrt(n)
        # (e.g. 300 threads, n=65701 -> chunk=220, 299*220 > n): the
        # trailing threads used to record negative counts, undersizing
        # the merge buffer (heap corruption). Needs a thread count that
        # can't be set in-process, so run in a subprocess — and verify
        # by OUTPUT, not exit code: the threadpool can segfault at
        # teardown after the work completed correctly.
        code = textwrap.dedent(
            """
            import numpy as np
            from numba_utils.parallel import parallel_topk
            n = 65701
            arr = np.random.default_rng(0).normal(0.0, 100.0, n)
            got = np.sort(parallel_topk(arr, 10))
            expected = np.sort(arr)[-10:]
            assert np.array_equal(got, expected)
            print("PARALLEL_TOPK_OK", flush=True)
            """
        )
        env = dict(
            os.environ, NUMBA_NUM_THREADS="300", NUMBA_UTILS_CACHE="0"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        assert "PARALLEL_TOPK_OK" in result.stdout, (
            result.stdout + result.stderr
        )


class TestChunkedReduce:
    def test_bit_exact_serial_vs_parallel(self):
        from numba_utils.random import philox_uniform

        @chunked_reduce
        def mc_sum(chunk_id, start, end):
            acc = 0.0
            for i in range(start, end):
                acc += philox_uniform(99, i)
            return acc

        for n_chunks in (1, 3, 16, 64):
            par = mc_sum(200_000, n_chunks)
            ser = mc_sum(200_000, n_chunks, parallel=False)
            assert par == ser  # bit-identical, not approx

    def test_result_reasonable(self):
        from numba_utils.random import philox_uniform

        @chunked_reduce
        def mc_sum(chunk_id, start, end):
            acc = 0.0
            for i in range(start, end):
                acc += philox_uniform(7, i)
            return acc

        n = 100_000
        assert abs(mc_sum(n, 8) / n - 0.5) < 0.01

    def test_empty_and_chunks_exceeding_items(self):
        @chunked_reduce
        def count(chunk_id, start, end):
            return float(end - start)

        assert count(0, 4) == 0.0
        assert count(5, 16) == 5.0  # more chunks than items
        assert count(5, 16, parallel=False) == 5.0

    def test_jitted_drivers_exposed(self):
        @chunked_reduce
        def count(chunk_id, start, end):
            return float(end - start)

        assert count.serial(10, 3) == 10.0
        assert count.parallel(10, 3) == 10.0

    def test_invalid_args_raise(self):
        @chunked_reduce
        def count(chunk_id, start, end):
            return 0.0

        with pytest.raises(ValueError):
            count(-1, 4)
        with pytest.raises(ValueError):
            count(10, 0)
        with pytest.raises(TypeError):
            chunked_reduce(42)

    def test_np_random_kernel_breaks_the_guarantee(self):
        # The negative case for the bit-exactness contract: a kernel
        # drawing from Numba's per-thread global RNG must NOT match
        # between drivers — each thread has its own stream. Only
        # demonstrable with real thread parallelism.
        from numba import get_num_threads

        if get_num_threads() < 2:
            pytest.skip("needs >= 2 threads to demonstrate divergence")

        from numba_utils.random import seed as nu_seed

        @chunked_reduce
        def rng_sum(chunk_id, start, end):
            acc = 0.0
            for _ in range(start, end):
                acc += np.random.random()
            return acc

        nu_seed(0)
        par = rng_sum(400_000, 64)
        nu_seed(0)
        ser = rng_sum(400_000, 64, parallel=False)
        assert par != ser


class TestParallelHistogramDegenerateArgs:
    def test_huge_bins_raise_before_padding_overflow(self):
        # near 2**63 the (bins + 7) padding arithmetic overflows int64
        # into a negative dimension, which parallel lowering turned
        # into an out-of-bounds write (0xC0000005) instead of a clean
        # allocation error
        arr = np.random.default_rng(0).random(SERIAL_THRESHOLD * 2)
        for bins in (2**63 - 1, 2**63 - 7, (1 << 30) + 1):
            with pytest.raises(ValueError):
                parallel_histogram(arr, bins, 0.0, 1.0)

    def test_degenerate_span_raises(self):
        arr = np.zeros(SERIAL_THRESHOLD * 2)
        with pytest.raises(ValueError):
            parallel_histogram(arr, 64, 0.0, 5e-324)

    def test_bins_cap_not_evaded_below_serial_threshold(self):
        # the same oversized bins must be rejected regardless of array
        # length — below the serial threshold the check used to be
        # skipped by delegating to the serial path first (issue #12)
        small = np.random.default_rng(0).random(1000)  # < SERIAL_THRESHOLD
        with pytest.raises(ValueError):
            parallel_histogram(small, 2**31, 0.0, 1.0)
