import numpy as np
import pytest
from numba import njit

from numba_utils import diagnostics, njit_fast, njit_parallel


def _make_compiled():
    @njit_fast
    def scaled_sum(arr):
        acc = 0.0
        for x in arr:
            acc += x
        return acc * 2.0

    scaled_sum(np.arange(10, dtype=np.float64))
    return scaled_sum


class TestInspect:
    def test_reports_flags_and_signatures(self):
        fn = _make_compiled()
        report = diagnostics.inspect(fn)
        assert report.name == "scaled_sum"
        assert len(report.signatures) == 1
        assert report.fastmath is True
        assert report.nogil is True
        assert report.parallel is False
        assert report.cache_enabled is True

    def test_uncompiled_has_no_signatures(self):
        @njit
        def untouched(x):
            return x

        report = diagnostics.inspect(untouched)
        assert report.signatures == ()

    def test_non_dispatcher_raises(self):
        with pytest.raises(TypeError):
            diagnostics.inspect(sorted)

    def test_report_is_immutable(self):
        report = diagnostics.inspect(_make_compiled())
        with pytest.raises(AttributeError):
            report.name = "other"


class TestShow:
    def test_contains_key_facts(self, capsys):
        fn = _make_compiled()
        text = diagnostics.show(fn, verbose=False)
        assert capsys.readouterr().out == ""
        assert "scaled_sum" in text
        assert "Cache: enabled" in text
        assert "Fastmath: yes" in text
        assert "Parallel: no" in text

    def test_verbose_prints(self, capsys):
        diagnostics.show(_make_compiled())
        assert "scaled_sum" in capsys.readouterr().out


class TestCheck:
    def test_cache_warning_with_recommendation(self):
        warnings = diagnostics.check(_make_compiled(), verbose=False)
        cache_warnings = [w for w in warnings if "NUMBA_UTILS_CACHE=0" in w]
        assert len(cache_warnings) == 1

    def test_fastmath_warning(self):
        warnings = diagnostics.check(_make_compiled(), verbose=False)
        assert any("fastmath" in w for w in warnings)

    def test_parallel_warning(self):
        @njit_parallel
        def pfn(arr):
            return arr.sum()

        warnings = diagnostics.check(pfn, verbose=False)
        assert any("prange" in w for w in warnings)

    def test_uncompiled_suggests_warmup(self):
        @njit
        def cold(x):
            return x

        warnings = diagnostics.check(cold, verbose=False)
        assert any("warmup" in w for w in warnings)

    def test_plain_njit_is_clean(self):
        @njit
        def plain(x):
            return x + 1

        plain(1.0)
        assert diagnostics.check(plain, verbose=False) == []

    def test_verbose_prints_warnings(self, capsys):
        diagnostics.check(_make_compiled())
        assert "diagnostics.check" in capsys.readouterr().out
