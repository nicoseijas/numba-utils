"""Head-to-head timing comparison between two callables."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import repeat
from time import perf_counter
from typing import Any, Callable

# Batched timing: one perf_counter pair per call costs ~100-200 ns of
# timer overhead, which inflates the MEAN of ns-scale kernels by
# 10-40% depending on the machine (the median is robust; the mean is
# not). Samples are therefore batches of `inner` back-to-back calls,
# auto-sized so one sample lasts ~100 µs — timer overhead drops below
# ~0.2% of the sample. Functions at or above ~100 µs per call keep
# inner=1: the overhead is already negligible there.
_TARGET_SAMPLE_S = 100e-6
_MAX_INNER = 65536


def _calibrate_inner(
    fn: Callable[..., Any], args: tuple, kwargs: dict
) -> int:
    """Batch size for one timed sample, from an uncounted probe call."""
    start = perf_counter()
    fn(*args, **kwargs)
    probe = perf_counter() - start
    if probe >= _TARGET_SAMPLE_S:
        return 1
    if probe <= 0.0:
        return _MAX_INNER
    return min(_MAX_INNER, max(1, math.ceil(_TARGET_SAMPLE_S / probe)))


def _time_batched(
    fn: Callable[..., Any], args: tuple, kwargs: dict, inner: int
) -> float:
    """One sample: mean per-call seconds over `inner` back-to-back calls."""
    start = perf_counter()
    for _ in repeat(None, inner):
        fn(*args, **kwargs)
    return (perf_counter() - start) / inner


@dataclass(frozen=True)
class TimingStats:
    """Per-callable timing statistics, in per-call seconds.

    Each of the ``runs`` samples is the mean over ``inner`` back-to-back
    calls (``inner=1`` means call-by-call timing). With ``inner > 1``,
    ``variance``/``minimum``/``maximum`` describe BATCH means — tighter
    than per-call spread by roughly a factor of ``inner``.
    """

    name: str
    runs: int
    mean: float
    median: float
    variance: float
    minimum: float
    maximum: float
    inner: int = 1

    @classmethod
    def from_times(
        cls, name: str, times: list[float], inner: int = 1
    ) -> "TimingStats":
        return cls(
            name=name,
            runs=len(times),
            mean=statistics.fmean(times),
            median=statistics.median(times),
            variance=statistics.variance(times) if len(times) > 1 else 0.0,
            minimum=min(times),
            maximum=max(times),
            inner=inner,
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Result of :func:`compare`. ``speedup`` > 1 means ``second`` is faster."""

    first: TimingStats
    second: TimingStats

    @property
    def speedup(self) -> float:
        """``first.mean / second.mean``. ``inf`` when the second mean
        is 0.0 (reachable: ``perf_counter`` has finite resolution, and
        a trivial kernel can time as 0.0 on every sample — especially
        on Windows), ``nan`` when both are 0."""
        if self.second.mean == 0.0:
            return float("inf") if self.first.mean > 0.0 else float("nan")
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
                f"{stats.variance * 1e6:>10.4f} ms^2"
            )
        lines.append(
            f"speedup ({self.second.name} vs {self.first.name}): "
            f"{self.speedup:.2f}x"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def compare(
    first: Callable[..., Any],
    second: Callable[..., Any],
    *,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    n: int = 100,
    warmup_runs: int = 1,
    inner: int | None = None,
) -> ComparisonResult:
    """Time ``first`` and ``second`` on identical inputs and compare.

    Both callables receive the same ``args``/``kwargs`` on every run.
    ``warmup_runs`` uncounted calls precede measurement so JIT compilation
    never pollutes the numbers. Inputs are not copied between runs — don't
    pass functions that mutate their arguments.

    Samples are INTERLEAVED: each of the ``n`` rounds times one sample
    of each callable, alternating which goes first — thermal drift,
    frequency scaling and cache state land on both, not on whichever
    ran second. Each sample is a batch of ``inner`` back-to-back calls
    (per-call seconds reported); ``inner=None`` auto-calibrates per
    callable so timer overhead cannot skew the mean of the faster one
    (see :class:`TimingStats`). Auto-calibration probes each callable
    once, uncounted, and needs ``warmup_runs >= 1``; with
    ``warmup_runs=0``, ``inner`` defaults to 1.
    """
    if not callable(first) or not callable(second):
        raise TypeError("compare() expects two callables")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if inner is not None and inner < 1:
        raise ValueError(f"inner must be >= 1, got {inner}")
    kwargs = kwargs or {}

    for fn in (first, second):
        for _ in range(warmup_runs):
            fn(*args, **kwargs)

    if inner is not None:
        inner_first = inner_second = inner
    elif warmup_runs > 0:
        inner_first = _calibrate_inner(first, args, kwargs)
        inner_second = _calibrate_inner(second, args, kwargs)
    else:
        inner_first = inner_second = 1

    times_first: list[float] = []
    times_second: list[float] = []
    for r in range(n):
        order = (
            ((first, inner_first, times_first), (second, inner_second, times_second))
            if r % 2 == 0
            else ((second, inner_second, times_second), (first, inner_first, times_first))
        )
        for fn, fn_inner, sink in order:
            sink.append(_time_batched(fn, args, kwargs, fn_inner))

    return ComparisonResult(
        first=TimingStats.from_times(_fn_name(first), times_first, inner_first),
        second=TimingStats.from_times(
            _fn_name(second), times_second, inner_second
        ),
    )


def _fn_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", repr(fn))
