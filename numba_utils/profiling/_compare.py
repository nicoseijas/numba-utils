"""Head-to-head timing comparison between two callables."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class TimingStats:
    """Per-callable timing statistics, in seconds."""

    name: str
    runs: int
    mean: float
    median: float
    variance: float
    minimum: float
    maximum: float

    @classmethod
    def from_times(cls, name: str, times: list[float]) -> "TimingStats":
        return cls(
            name=name,
            runs=len(times),
            mean=statistics.fmean(times),
            median=statistics.median(times),
            variance=statistics.variance(times) if len(times) > 1 else 0.0,
            minimum=min(times),
            maximum=max(times),
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Result of :func:`compare`. ``speedup`` > 1 means ``second`` is faster."""

    first: TimingStats
    second: TimingStats

    @property
    def speedup(self) -> float:
        return self.first.mean / self.second.mean

    def summary(self) -> str:
        lines = [
            f"{'':<12}{'mean':>12}{'median':>12}{'variance':>12}",
        ]
        for stats in (self.first, self.second):
            lines.append(
                f"{stats.name:<12}"
                f"{stats.mean * 1e3:>10.3f} ms"
                f"{stats.median * 1e3:>10.3f} ms"
                f"{stats.variance * 1e6:>10.4f} ms²"
            )
        lines.append(
            f"speedup ({self.second.name} vs {self.first.name}): "
            f"{self.speedup:.2f}x"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _time_runs(
    fn: Callable[..., Any], args: tuple, kwargs: dict, n: int
) -> list[float]:
    times = []
    for _ in range(n):
        start = perf_counter()
        fn(*args, **kwargs)
        times.append(perf_counter() - start)
    return times


def compare(
    first: Callable[..., Any],
    second: Callable[..., Any],
    *,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    n: int = 100,
    warmup_runs: int = 1,
) -> ComparisonResult:
    """Time ``first`` and ``second`` on identical inputs and compare.

    Both callables receive the same ``args``/``kwargs`` on every run.
    ``warmup_runs`` uncounted calls precede measurement so JIT compilation
    never pollutes the numbers. Inputs are not copied between runs — don't
    pass functions that mutate their arguments.
    """
    if not callable(first) or not callable(second):
        raise TypeError("compare() expects two callables")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    kwargs = kwargs or {}

    for fn in (first, second):
        for _ in range(warmup_runs):
            fn(*args, **kwargs)

    return ComparisonResult(
        first=TimingStats.from_times(
            _fn_name(first), _time_runs(first, args, kwargs, n)
        ),
        second=TimingStats.from_times(
            _fn_name(second), _time_runs(second, args, kwargs, n)
        ),
    )


def _fn_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", repr(fn))
