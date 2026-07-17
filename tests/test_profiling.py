import time

import numpy as np
import pytest
from numba import njit

from numba_utils.profiling import (
    BenchmarkResult,
    ComparisonResult,
    benchmark,
    compare,
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
