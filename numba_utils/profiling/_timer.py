"""Block timing, function benchmarking, JIT warmup and compile stats.

The function-mode :func:`benchmark` is correct for JIT code BY DEFAULT:
uncounted warmup calls run first, so compilation never pollutes the
measurement — the single most common mistake in Numba benchmarks found
in the wild (see docs/benchmarking.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from numba_utils.profiling._compare import TimingStats


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
) -> Any:
    """Benchmark a function correctly, or time a block.

    **Function mode** — pass a callable::

        stats = benchmark(foo, args=(arr,), n=50)

    Runs ``warmup_runs`` uncounted calls first (JIT compilation and any
    lazy setup are excluded from the measurement — the correct default
    for jitted code), then times ``n`` calls and returns
    :class:`TimingStats` (mean/median/variance/min/max). Set
    ``warmup_runs=0`` only if you explicitly want compilation included.
    ``args``/``kwargs`` are NOT copied between runs — don't pass a
    function that mutates its arguments.

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
        for _ in range(warmup_runs):
            target(*args, **kwargs)
        times = []
        for _ in range(n):
            start = perf_counter()
            target(*args, **kwargs)
            times.append(perf_counter() - start)
        name = getattr(target, "__name__", repr(target))
        stats = TimingStats.from_times(name, times)
        if verbose:
            print(
                f"[{name}] mean {stats.mean * 1e3:.3f} ms | "
                f"median {stats.median * 1e3:.3f} ms | "
                f"runs {stats.runs} (after {warmup_runs} warmup)"
            )
        return stats
    label = target if isinstance(target, str) else "block"
    return _BenchmarkBlock(label, verbose=verbose)


def warmup(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> float:
    """Call ``fn`` once to trigger JIT compilation; returns the elapsed seconds."""
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__!r}")
    start = perf_counter()
    fn(*args, **kwargs)
    return perf_counter() - start


def compile_time(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> float:
    """Estimate JIT compilation time in seconds for ``fn`` with these arguments.

    Measures the first call (compile + run) minus a second call (run only).
    Returns ~0.0 if ``fn`` was already compiled for this signature.
    """
    if not callable(fn):
        raise TypeError(f"expected a callable, got {type(fn).__name__!r}")
    start = perf_counter()
    fn(*args, **kwargs)
    first = perf_counter() - start

    start = perf_counter()
    fn(*args, **kwargs)
    second = perf_counter() - start
    return max(first - second, 0.0)


def compile_stats(fn: Callable[..., Any]):
    """Compilation report for a dispatcher: signatures, per-signature
    compile times, cache state and flags.

    Thin alias for :func:`numba_utils.diagnostics.inspect`; returns its
    immutable ``FunctionReport``.
    """
    from numba_utils.diagnostics import inspect

    return inspect(fn)
