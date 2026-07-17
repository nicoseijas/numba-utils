import numpy as np
import pytest

from numba_utils.decorators import (
    DEV_MODE_ENV_VAR,
    boundscheck,
    cached_njit,
    njit_fast,
    parallel,
)


def _sum_impl(arr):
    acc = 0.0
    for x in arr:
        acc += x
    return acc


class TestNjitFast:
    def test_bare_form_computes_correctly(self):
        fn = njit_fast(_sum_impl)
        arr = np.arange(100, dtype=np.float64)
        assert fn(arr) == pytest.approx(arr.sum())

    def test_called_form_computes_correctly(self):
        fn = njit_fast()(_sum_impl)
        arr = np.arange(100, dtype=np.float64)
        assert fn(arr) == pytest.approx(arr.sum())

    def test_defaults_applied(self):
        fn = njit_fast(_sum_impl)
        assert fn.targetoptions["fastmath"] is True
        assert fn.targetoptions["nogil"] is True

    def test_override_wins_over_default(self):
        fn = njit_fast(fastmath=False)(_sum_impl)
        assert fn.targetoptions["fastmath"] is False

    def test_is_numba_dispatcher(self):
        fn = njit_fast(_sum_impl)
        assert hasattr(fn, "py_func")

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            njit_fast(42)


class TestParallel:
    def test_computes_correctly(self):
        from numba import prange

        @parallel
        def psum(arr):
            acc = 0.0
            for i in prange(arr.shape[0]):
                acc += arr[i]
            return acc

        arr = np.ones(10_000, dtype=np.float64)
        assert psum(arr) == pytest.approx(10_000.0)

    def test_parallel_flag_set(self):
        fn = parallel(_sum_impl)
        assert fn.targetoptions["parallel"] is True


class TestCachedNjit:
    def test_computes_correctly(self):
        fn = cached_njit(_sum_impl)
        arr = np.arange(50, dtype=np.float64)
        assert fn(arr) == pytest.approx(arr.sum())


class TestBoundscheck:
    def test_production_mode_compiles_and_runs(self, monkeypatch):
        monkeypatch.delenv(DEV_MODE_ENV_VAR, raising=False)

        @boundscheck
        def get(arr, i):
            return arr[i]

        arr = np.arange(10, dtype=np.float64)
        assert get(arr, 3) == 3.0

    def test_dev_mode_raises_on_out_of_bounds(self, monkeypatch):
        monkeypatch.setenv(DEV_MODE_ENV_VAR, "1")

        @boundscheck
        def get(arr, i):
            return arr[i]

        arr = np.arange(10, dtype=np.float64)
        assert get(arr, 3) == 3.0
        with pytest.raises(IndexError):
            get(arr, 10)
