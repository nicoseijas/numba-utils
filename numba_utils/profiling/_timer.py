"""Block timing, function benchmarking, JIT warmup and compile stats.

The function-mode :func:`benchmark` is correct for JIT code BY DEFAULT:
uncounted warmup calls run first, so compilation never pollutes the
measurement — the single most common mistake in Numba benchmarks found
in the wild (see docs/benchmarking.md).
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from numba_utils.profiling._compare import (
    TimingStats,
    _calibrate_inner,
    _time_batched,
)


def _cache_hits(fn: Any) -> int:
    try:
        return int(sum(fn.stats.cache_hits.values()))
    except (AttributeError, TypeError):
        return 0


@dataclass(frozen=True)
class BenchmarkResult:
    """Elapsed wall-clock time of a timed block, in seconds."""

    label: str
    elapsed: float

    def __str__(self) -> str:
        return f"[{self.label}] {self.elapsed * 1e3:.3f} ms"


class _BenchmarkBlock:
    """Context manager timing a block with ``perf_counter``."""

    def __init__(self, label: str = "block", *, verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.result: BenchmarkResult | None = None
        self._start: float | None = None

    def __enter__(self) -> "_BenchmarkBlock":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = perf_counter() - self._start  # type: ignore[operand]
        self.result = BenchmarkResult(label=self.label, elapsed=elapsed)
        if self.verbose and exc_type is None:
            print(self.result)


def benchmark(
    target: Callable[..., Any] | str | None = None,
    *,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    n: int = 100,
    warmup_runs: int = 1,
    verbose: bool = True,
    inner: int | None = None,
) -> Any:
    """Benchmark a function correctly, or time a block.

    **Function mode** — pass a callable::

        stats = benchmark(foo, args=(arr,), n=50)

    Runs ``warmup_runs`` uncounted calls first (JIT compilation and any
    lazy setup are excluded from the measurement — the correct default
    for jitted code), then takes ``n`` timed samples and returns
    :class:`TimingStats` (per-call mean/median/variance/min/max). Set
    ``warmup_runs=0`` only if you explicitly want compilation included.
    ``args``/``kwargs`` are NOT copied between runs — don't pass a
    function that mutates its arguments.

    Each sample is a batch of ``inner`` back-to-back calls: two
    ``perf_counter`` reads per CALL cost ~100-200 ns of timer overhead,
    inflating the measured MEAN of ns-scale kernels by 10-40%
    machine-dependent (the median is robust to it; the mean is not).
    ``inner=None`` auto-calibrates from one uncounted probe call so a
    sample lasts ~100 µs (functions at or above that per call keep
    ``inner=1``); pass ``inner=1`` to force call-by-call timing.
    Auto-calibration needs ``warmup_runs >= 1`` — with
    ``warmup_runs=0`` the probe would swallow the compilation you asked
    to measure, so ``inner`` defaults to 1 there.

    **Block mode** — pass a label (or nothing) and use as a context
    manager::

        with benchmark("hot loop") as b:
            foo()
        print(b.result.elapsed)

    Block mode measures whatever ran, compilation included — call
    :func:`warmup` first when that's not what you want.
    """
    if callable(target):
        kwargs = kwargs or {}
        if n < 1:
            raise ValueError(f"benchmark: n must be >= 1, got {n}")
        if warmup_runs < 0:
            raise ValueError(
                f"benchmark: warmup_runs must be >= 0, got {warmup_runs}"
            )
        if inner is not None and inner < 1:
            raise ValueError(f"benchmark: inner must be >= 1, got {inner}")
        for _ in range(warmup_runs):
            target(*args, **kwargs)
        if inner is None:
            inner = (
                _calibrate_inner(target, args, kwargs)
                if warmup_runs > 0
                else 1
            )
        times = [_time_batched(target, args, kwargs, inner) for _ in range(n)]
        name = getattr(target, "__name__", repr(target))
        stats = TimingStats.from_times(name, times, inner)
        if verbose:
            batch = f" x{inner} calls/sample" if inner > 1 else ""
            print(
                f"[{name}] mean {stats.mean * 1e3:.3f} ms | "
                f"median {stats.median * 1e3:.3f} ms | "
                f"runs {stats.runs}{batch} (after {warmup_runs} warmup)"
            )
        return stats
    label = target if isinstance(target, str) else "block"
    return _BenchmarkBlock(label, verbose=verbose)


def warmup(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> float:
    """Call ``fn`` once to trigger JIT compilation; returns the elapsed seconds.

    One call warms exactly ONE signature: Numba compiles per argument
    dtype/rank combination, so a function later called with float32 AND
    float64 arrays needs a warmup per combination — passing a list of
    argument sets here is a ``TypingError``, not a multi-warmup. For
    several signatures, use :func:`warmup_signatures`.
    """
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__!r}")
    start = perf_counter()
    fn(*args, **kwargs)
    return perf_counter() - start


def warmup_signatures(
    fn: Callable[..., Any], arg_sets: Any, /
) -> list[float]:
    """One :func:`warmup` call per argument tuple in ``arg_sets``;
    returns the elapsed seconds of each.

    Numba compiles per SIGNATURE, so each dtype/rank combination the
    function will serve needs its own warmup call::

        warmup_signatures(fn, [(arr_f32,), (arr_f64,)])

    Each element must be a TUPLE of positional arguments — wrap
    single-argument calls as ``(arr,)``.
    """
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__!r}")
    elapsed = []
    for args in arg_sets:
        if not isinstance(args, tuple):
            raise TypeError(
                "warmup_signatures: each element of arg_sets must be a "
                f"tuple of positional arguments, got {type(args).__name__!r}"
                " — wrap single arguments as (arg,)"
            )
        elapsed.append(warmup(fn, *args))
    return elapsed


def compile_time(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> float:
    """Estimate JIT compilation time in seconds for ``fn`` with these arguments.

    Measures the first call (compile + run) minus a second call (run only).
    Returns ~0.0 if ``fn`` was already compiled for this signature.

    Caveat — warm on-disk cache: with ``cache=True`` and a populated
    ``__pycache__``, the first call LOADS the compiled binary instead
    of compiling; the measured time is then cache-load time, which can
    be ~40x smaller than a true cold compile. When that is detected
    (the dispatcher's cache-hit counter increased during the call), a
    ``RuntimeWarning`` says so — to measure real compilation, delete
    ``__pycache__`` or compile with ``cache=False``.
    """
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__!r}")
    hits_before = _cache_hits(fn)
    start = perf_counter()
    fn(*args, **kwargs)
    first = perf_counter() - start

    start = perf_counter()
    fn(*args, **kwargs)
    second = perf_counter() - start
    if _cache_hits(fn) > hits_before:
        _warnings.warn(
            "compile_time: the first call loaded a cached binary from "
            "disk — this measured cache-load time, NOT compilation "
            "(can be ~40x smaller). Delete __pycache__ or use "
            "cache=False to measure a true cold compile.",
            RuntimeWarning,
            stacklevel=2,
        )
    return max(first - second, 0.0)


def compile_stats(fn: Callable[..., Any]):
    """Compilation report for a dispatcher: signatures, per-signature
    compile times, cache state and flags.

    Thin alias for :func:`numba_utils.diagnostics.inspect`; returns its
    immutable ``FunctionReport``.
    """
    from numba_utils.diagnostics import inspect

    return inspect(fn)
