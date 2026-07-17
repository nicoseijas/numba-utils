import numpy as np
import pytest
from numba import njit

from numba_utils import choice, cumulative_sum, insertion_sort
from numba_utils.testing import (
    assert_close,
    assert_equivalent,
    deterministic_rng,
    random_arrays,
)


def _py_cumsum(arr):
    return np.cumsum(arr)


class TestAssertClose:
    def test_passes_on_close(self):
        assert_close(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-12]))

    def test_fails_on_far(self):
        with pytest.raises(AssertionError):
            assert_close(1.0, 1.1)


class TestAssertEquivalent:
    def test_validates_matching_kernel(self):
        checked = assert_equivalent(
            _py_cumsum,
            cumulative_sum,
            random_arrays(n_cases=10, size=500, seed=1),
        )
        assert checked == 15  # 10 random + 5 edge cases

    def test_protects_against_mutation(self):
        # insertion_sort mutates its input; np.sort does not. Per-call
        # copies mean the mutating candidate still compares correctly.
        assert_equivalent(
            np.sort,
            insertion_sort,
            random_arrays(n_cases=3, size=50, seed=2),
        )

    def test_reports_failing_case(self):
        def wrong(arr):
            return np.cumsum(arr) + 1.0

        with pytest.raises(AssertionError, match="case 0"):
            assert_equivalent(
                _py_cumsum, wrong, random_arrays(n_cases=1, size=10)
            )

    def test_tuple_results_compared_elementwise(self):
        def ref(arr):
            return arr.min(), arr.max()

        @njit
        def cand(arr):
            lo = arr[0]
            hi = arr[0]
            for x in arr:
                if x < lo:
                    lo = x
                elif x > hi:
                    hi = x
            return lo, hi

        assert_equivalent(ref, cand, random_arrays(n_cases=5, size=100))

    def test_empty_inputs_raise(self):
        with pytest.raises(AssertionError, match="no cases"):
            assert_equivalent(_py_cumsum, cumulative_sum, [])

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            assert_equivalent(42, cumulative_sum, [np.ones(3)])


class TestRandomArrays:
    def test_deterministic_per_seed(self):
        a = list(random_arrays(n_cases=3, size=20, seed=5))
        b = list(random_arrays(n_cases=3, size=20, seed=5))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_edge_cases_present(self):
        cases = list(random_arrays(n_cases=0, size=10))
        assert len(cases) == 5
        assert (cases[0] == cases[0][0]).all()  # constant
        assert (np.diff(cases[1]) >= 0).all()  # ascending
        assert (np.diff(cases[2]) <= 0).all()  # descending
        assert cases[4].shape == (1,)  # single element

    def test_dtype_respected(self):
        for arr in random_arrays(n_cases=2, size=8, dtype=np.int32):
            assert arr.dtype == np.int32

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            list(random_arrays(n_cases=-1, size=10))
        with pytest.raises(ValueError):
            list(random_arrays(n_cases=1, size=0))


class TestDeterministicRng:
    def test_pins_all_three_worlds(self):
        gen_a = deterministic_rng(11)
        legacy_a = np.random.random(3)
        modern_a = gen_a.random(3)
        numba_a = choice(np.arange(100, dtype=np.float64), 10)

        gen_b = deterministic_rng(11)
        np.testing.assert_array_equal(np.random.random(3), legacy_a)
        np.testing.assert_array_equal(gen_b.random(3), modern_a)
        np.testing.assert_array_equal(
            choice(np.arange(100, dtype=np.float64), 10), numba_a
        )
