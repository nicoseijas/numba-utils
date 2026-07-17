import numpy as np
import pytest

from numba_utils import cached_njit, config, configure, njit_fast


def _sum_impl(arr):
    acc = 0.0
    for x in arr:
        acc += x
    return acc


def _caching_enabled(dispatcher) -> bool:
    return type(dispatcher._cache).__name__ != "NullCache"


class TestConfigure:
    def test_cache_off_via_code(self):
        configure(cache=False)
        assert not _caching_enabled(cached_njit(_sum_impl))

    def test_attribute_style(self):
        config.cache = False
        assert not _caching_enabled(cached_njit(_sum_impl))

    def test_global_override_beats_per_call_argument(self):
        configure(fastmath=False)
        fn = njit_fast(fastmath=True)(_sum_impl)
        assert fn.targetoptions["fastmath"] is False

    def test_parallel_forced_globally(self):
        configure(parallel=True)
        fn = cached_njit(_sum_impl)
        assert fn.targetoptions["parallel"] is True

    def test_reset_restores_defaults(self):
        configure(cache=False)
        config.reset()
        assert _caching_enabled(cached_njit(_sum_impl))

    def test_clear_single_option_with_none(self):
        configure(cache=False)
        configure(cache=None)
        assert _caching_enabled(cached_njit(_sum_impl))

    def test_unknown_option_raises(self):
        with pytest.raises(ValueError):
            configure(cachee=False)

    def test_non_bool_value_raises(self):
        with pytest.raises(ValueError):
            configure(cache="no")

    def test_decorated_function_still_computes(self):
        configure(cache=False, fastmath=False)
        fn = njit_fast(_sum_impl)
        arr = np.arange(20, dtype=np.float64)
        assert fn(arr) == pytest.approx(arr.sum())


class TestEnvironmentFallback:
    def test_env_used_when_code_level_unset(self, monkeypatch):
        monkeypatch.setenv("NUMBA_UTILS_FASTMATH", "0")
        fn = njit_fast(_sum_impl)
        assert fn.targetoptions["fastmath"] is False

    def test_code_level_beats_environment(self, monkeypatch):
        monkeypatch.setenv("NUMBA_UTILS_CACHE", "1")
        configure(cache=False)
        assert not _caching_enabled(cached_njit(_sum_impl))

    def test_garbage_env_value_means_unset(self, monkeypatch):
        monkeypatch.setenv("NUMBA_UTILS_CACHE", "maybe")
        assert _caching_enabled(cached_njit(_sum_impl))
