"""Block timing, JIT warmup and compile-time measurement."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkResult:
    """Elapsed wall-clock time of a timed block, in seconds."""

    label: str
    elapsed: float

    def __str__(self) -> str:
        return f"[{self.label}] {self.elapsed * 1e3:.3f} ms"


class benchmark:
    """Context manager that times a block with ``perf_counter``.

    ::

        with benchmark("hot loop") as b:
            foo()
        print(b.result.elapsed)

    Prints the result on exit unless ``verbose=False``. The measurement
    includes JIT compilation if the callee wasn't warmed up — call
    :func:`warmup` first when that's not what you want.
    """

    def __init__(self, label: str = "block", *, verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.result: BenchmarkResult | None = None
        self._start: float | None = None

    def __enter__(self) -> "benchmark":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = perf_counter() - self._start  # type: ignore[operand]
        self.result = BenchmarkResult(label=self.label, elapsed=elapsed)
        if self.verbose and exc_type is None:
            print(self.result)


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
