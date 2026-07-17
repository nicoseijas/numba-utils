import numpy as np
import pytest
from numba import njit

from numba_utils.algorithms import (
    argmax2,
    counting_sort,
    fast_argpartition,
    insertion_sort,
    nth_element,
    partial_sort,
    quickselect,
    radix_sort,
    topk,
)

RNG = np.random.default_rng(123)


class TestNthElement:
    def test_places_kth_smallest(self):
        arr = RNG.normal(0, 1, 500)
        expected = np.sort(arr)
        for k in [0, 1, 250, 498, 499]:
            work = arr.copy()
            value = nth_element(work, k)
            assert value == expected[k]
            assert work[k] == expected[k]
            assert work[:k].max(initial=-np.inf) <= work[k]
            assert work[k + 1 :].min(initial=np.inf) >= work[k]

    def test_mutates_in_place(self):
        arr = np.array([3.0, 1.0, 2.0])
        nth_element(arr, 0)
        assert arr[0] == 1.0

    def test_duplicates(self):
        arr = np.full(50, 5.0)
        assert nth_element(arr, 25) == 5.0

    def test_errors(self):
        with pytest.raises(ValueError):
            nth_element(np.empty(0), 0)
        with pytest.raises(ValueError):
            nth_element(np.ones(3), 3)
        with pytest.raises(ValueError):
            nth_element(np.ones(3), -1)


class TestQuickselect:
    def test_matches_sorted(self):
        arr = RNG.normal(0, 1, 1000)
        expected = np.sort(arr)
        for k in [0, 10, 500, 999]:
            assert quickselect(arr, k) == expected[k]

    def test_input_not_mutated(self):
        arr = RNG.normal(0, 1, 100)
        original = arr.copy()
        quickselect(arr, 50)
        np.testing.assert_array_equal(arr, original)


class TestFastArgpartition:
    def test_returns_k_smallest_indices(self):
        arr = RNG.normal(0, 1, 500)
        for k in [1, 5, 250, 500]:
            idx = fast_argpartition(arr, k)
            assert idx.shape == (k,)
            np.testing.assert_allclose(
                np.sort(arr[idx]), np.sort(arr)[:k]
            )

    def test_input_not_mutated(self):
        arr = RNG.normal(0, 1, 100)
        original = arr.copy()
        fast_argpartition(arr, 10)
        np.testing.assert_array_equal(arr, original)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            fast_argpartition(np.ones(3), 0)
        with pytest.raises(ValueError):
            fast_argpartition(np.ones(3), 4)


class TestTopk:
    def test_matches_numpy(self):
        arr = RNG.normal(0, 1, 1000)
        for k in [1, 10, 999, 1000]:
            expected = np.sort(arr)[-k:][::-1]
            np.testing.assert_allclose(topk(arr, k), expected)

    def test_input_not_mutated(self):
        arr = RNG.normal(0, 1, 100)
        original = arr.copy()
        topk(arr, 5)
        np.testing.assert_array_equal(arr, original)

    def test_int_dtype(self):
        arr = np.array([5, 1, 9, 3], dtype=np.int64)
        np.testing.assert_array_equal(topk(arr, 2), [9, 5])

    def test_callable_from_jitted_code(self):
        @njit
        def best_two_sum(arr):
            t = topk(arr, 2)
            return t[0] + t[1]

        assert best_two_sum(np.array([1.0, 5.0, 3.0, 4.0])) == 9.0


class TestArgmax2:
    def test_index_and_value(self):
        arr = RNG.normal(0, 1, 1000)
        idx, value = argmax2(arr)
        assert idx == np.argmax(arr)
        assert value == arr.max()

    def test_first_occurrence_on_ties(self):
        idx, value = argmax2(np.array([1.0, 7.0, 7.0]))
        assert idx == 1 and value == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            argmax2(np.empty(0))


class TestInsertionSort:
    def test_sorts_in_place(self):
        arr = RNG.normal(0, 1, 200)
        expected = np.sort(arr)
        result = insertion_sort(arr)
        assert result is arr
        np.testing.assert_array_equal(arr, expected)

    def test_edge_cases(self):
        np.testing.assert_array_equal(insertion_sort(np.empty(0)), np.empty(0))
        np.testing.assert_array_equal(insertion_sort(np.array([1.0])), [1.0])


class TestPartialSort:
    def test_front_k_sorted(self):
        arr = RNG.normal(0, 1, 500)
        expected = np.sort(arr)
        for k in [1, 10, 499, 500]:
            work = arr.copy()
            partial_sort(work, k)
            np.testing.assert_array_equal(work[:k], expected[:k])
            assert sorted(work.tolist()) == sorted(arr.tolist())

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            partial_sort(np.ones(3), 0)


class TestCountingSort:
    def test_matches_numpy(self):
        arr = RNG.integers(-50, 50, 10_000).astype(np.int64)
        np.testing.assert_array_equal(counting_sort(arr), np.sort(arr))

    def test_input_not_mutated(self):
        arr = np.array([3, 1, 2], dtype=np.int64)
        counting_sort(arr)
        np.testing.assert_array_equal(arr, [3, 1, 2])

    def test_int32(self):
        arr = RNG.integers(0, 100, 1000).astype(np.int32)
        result = counting_sort(arr)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, np.sort(arr))

    def test_huge_range_raises(self):
        with pytest.raises(ValueError):
            counting_sort(np.array([0, 2**40], dtype=np.int64))


class TestRadixSort:
    def test_int64_with_negatives(self):
        arr = RNG.integers(-(2**62), 2**62, 10_000).astype(np.int64)
        np.testing.assert_array_equal(radix_sort(arr), np.sort(arr))

    def test_int32(self):
        arr = RNG.integers(-(2**31), 2**31, 10_000).astype(np.int32)
        result = radix_sort(arr)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, np.sort(arr))

    def test_uint64_above_int64_range(self):
        arr = np.array([2**63 + 5, 3, 2**64 - 1, 0], dtype=np.uint64)
        np.testing.assert_array_equal(radix_sort(arr), np.sort(arr))

    def test_small_range_early_exit(self):
        arr = RNG.integers(0, 200, 10_000).astype(np.int64)
        np.testing.assert_array_equal(radix_sort(arr), np.sort(arr))

    def test_input_not_mutated(self):
        arr = np.array([3, 1, 2], dtype=np.int64)
        radix_sort(arr)
        np.testing.assert_array_equal(arr, [3, 1, 2])

    def test_edge_cases(self):
        assert radix_sort(np.empty(0, dtype=np.int64)).shape == (0,)
        np.testing.assert_array_equal(
            radix_sort(np.array([7], dtype=np.int64)), [7]
        )
