import numpy as np
import pytest

from numba_utils.decorators import (
    CACHE_ENV_VAR,
    DEV_MODE_ENV_VAR,
    boundscheck,
    cached_njit,
    njit_fast,
    njit_parallel,
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

        @njit_parallel
        def psum(arr):
            acc = 0.0
            for i in prange(arr.shape[0]):
                acc += arr[i]
            return acc

        arr = np.ones(10_000, dtype=np.float64)
        assert psum(arr) == pytest.approx(10_000.0)

    def test_parallel_flag_set(self):
        fn = njit_parallel(_sum_impl)
        assert fn.targetoptions["parallel"] is True


class TestCachedNjit:
    def test_computes_correctly(self):
        fn = cached_njit(_sum_impl)
        arr = np.arange(50, dtype=np.float64)
        assert fn(arr) == pytest.approx(arr.sum())


def _caching_enabled(dispatcher) -> bool:
    return type(dispatcher._cache).__name__ != "NullCache"


class TestCacheKillSwitch:
    def test_default_has_caching(self, monkeypatch):
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
        fn = cached_njit(_sum_impl)
        assert _caching_enabled(fn)

    def test_kill_switch_disables_all_decorators(self, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, "0")
        for decorator in (cached_njit, njit_fast, njit_parallel, boundscheck):
            fn = decorator(_sum_impl)
            assert not _caching_enabled(fn)

    def test_kill_switch_overrides_explicit_cache_true(self, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, "0")
        fn = cached_njit(cache=True)(_sum_impl)
        assert not _caching_enabled(fn)

    def test_disabled_function_still_computes(self, monkeypatch):
        monkeypatch.setenv(CACHE_ENV_VAR, "off")
        fn = njit_fast(_sum_impl)
        arr = np.arange(10, dtype=np.float64)
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

    def test_dev_mode_cache_lock_beats_global_override(self, monkeypatch):
        # Numba's cache key ignores boundscheck, so a shared on-disk
        # cache would poison both directions (prod loads checked
        # binaries; dev loads unchecked ones and silently checks
        # nothing). The dev build must never touch the cache — not
        # even under a global cache=True override or an explicit
        # per-call argument.
        monkeypatch.setenv(DEV_MODE_ENV_VAR, "1")
        monkeypatch.setenv(CACHE_ENV_VAR, "1")

        fn = boundscheck(_sum_impl)
        assert type(fn._cache).__name__ == "NullCache"
        fn_explicit = boundscheck(cache=True)(_sum_impl)
        assert type(fn_explicit._cache).__name__ == "NullCache"
