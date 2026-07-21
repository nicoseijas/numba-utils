import time

import numpy as np
import pytest
from numba import njit

from numba_utils.profiling import (
    BenchmarkResult,
    ComparisonResult,
    TimingStats,
    benchmark,
    compare,
    compile_stats,
    compile_time,
    warmup,
)


class TestBenchmark:
    def test_measures_elapsed_time(self):
        with benchmark("sleep", verbose=False) as b:
            time.sleep(0.02)
        assert b.result is not None
        assert b.result.elapsed >= 0.01

    def test_result_is_immutable(self):
        with benchmark(verbose=False) as b:
            pass
        with pytest.raises(AttributeError):
            b.result.elapsed = 0.0

    def test_verbose_prints(self, capsys):
        with benchmark("labelled"):
            pass
        assert "labelled" in capsys.readouterr().out

    def test_str_format(self):
        r = BenchmarkResult(label="x", elapsed=0.001)
        assert "x" in str(r) and "ms" in str(r)


class TestBenchmarkFunctionMode:
    def test_returns_stats_with_n_runs(self):
        stats = benchmark(sorted, args=([3, 1, 2],), n=7, verbose=False)
        assert isinstance(stats, TimingStats)
        assert stats.runs == 7
        assert stats.mean > 0

    def test_warmup_excludes_compilation_by_default(self):
        @njit
        def fresh(x):
            return x * 2.0

        stats = benchmark(fresh, args=(1.0,), n=5, verbose=False)
        assert stats.mean < 0.005

    def test_verbose_prints_summary(self, capsys):
        benchmark(sorted, args=([1],), n=2)
        out = capsys.readouterr().out
        assert "sorted" in out and "mean" in out

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            benchmark(sorted, args=([1],), n=0)
        with pytest.raises(ValueError):
            benchmark(sorted, args=([1],), warmup_runs=-1)


class TestCompileStats:
    def test_reports_signatures_and_flags(self):
        @njit
        def doubled(x):
            return x * 2.0

        doubled(1.0)
        report = compile_stats(doubled)
        assert len(report.signatures) == 1
        assert report.parallel is False

    def test_non_dispatcher_raises(self):
        with pytest.raises(TypeError):
            compile_stats(sorted)


class TestWarmup:
    def test_compiles_function(self):
        @njit
        def double(x):
            return x * 2.0

        assert double.signatures == []
        elapsed = warmup(double, 1.0)
        assert elapsed > 0
        assert len(double.signatures) == 1

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            warmup(42)


class TestCompileTime:
    def test_fresh_function_has_positive_compile_time(self):
        @njit
        def triple(x):
            return x * 3.0

        assert compile_time(triple, 1.0) > 0

    def test_already_compiled_is_near_zero(self):
        @njit
        def quad(x):
            return x * 4.0

        warmup(quad, 1.0)
        assert compile_time(quad, 1.0) < 0.01

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            compile_time("not a function")


class TestCompare:
    def test_returns_stats_for_both(self):
        arr = np.arange(1000, dtype=np.float64)

        @njit
        def jit_sum(a):
            acc = 0.0
            for x in a:
                acc += x
            return acc

        result = compare(np.sum, jit_sum, args=(arr,), n=10)
        assert isinstance(result, ComparisonResult)
        assert result.first.runs == 10
        assert result.second.runs == 10
        assert result.first.mean > 0
        assert result.second.mean > 0
        assert result.speedup > 0

    def test_summary_mentions_both_names(self):
        result = compare(sorted, list, args=([3, 1, 2],), n=5)
        text = result.summary()
        assert "sorted" in text and "list" in text and "speedup" in text

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            compare(sorted, list, args=([1],), n=0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError):
            compare(sorted, list, args=([1],), warmup_runs=-1)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            compare(42, sorted)


class TestSpeedupDegenerateTimings:
    def test_zero_second_mean_gives_inf_not_crash(self):
        # perf_counter has finite resolution: a trivial kernel can
        # time as 0.0 on every sample (especially on Windows), and
        # speedup/summary() must survive that, not ZeroDivisionError
        from numba_utils.profiling import ComparisonResult, TimingStats

        zero = TimingStats("b", 3, 0.0, 0.0, 0.0, 0.0, 0.0)
        slow = TimingStats("a", 3, 1e-3, 1e-3, 0.0, 1e-3, 1e-3)
        result = ComparisonResult(first=slow, second=zero)
        assert result.speedup == float("inf")
        assert "inf" in result.summary()
        both_zero = ComparisonResult(first=zero, second=zero)
        assert np.isnan(both_zero.speedup)
