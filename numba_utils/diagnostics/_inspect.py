"""What did Numba actually build? Dispatcher introspection + known-issue checks.

Some report fields depend on Numba internals that vary across versions
(compile timers, cache statistics); those degrade to empty/zero rather
than failing, so a report is always produced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

_SIGNATURE_CHURN_THRESHOLD = 5


def _require_dispatcher(fn: Any) -> None:
    if not (
        hasattr(fn, "py_func")
        and hasattr(fn, "signatures")
        and hasattr(fn, "targetoptions")
    ):
        raise TypeError(
            "diagnostics: expected a Numba dispatcher (a function decorated "
            "with @njit or a numba-utils decorator), got "
            f"{type(fn).__name__!r}"
        )


@dataclass(frozen=True)
class FunctionReport:
    """Snapshot of a dispatcher's compilation state. All times in seconds."""

    name: str
    signatures: tuple[str, ...]
    cache_enabled: bool
    cache_hits: int
    cache_misses: int
    parallel: bool
    fastmath: bool
    nogil: bool
    boundscheck: bool
    compile_times_s: tuple[float, ...]
    asm_text_bytes: tuple[int, ...]


def inspect(fn: Any) -> FunctionReport:
    """Build a :class:`FunctionReport` for a compiled (or not yet compiled)
    dispatcher. Raises ``TypeError`` for anything that isn't one."""
    _require_dispatcher(fn)
    opts = fn.targetoptions
    hits = 0
    misses = 0
    try:
        stats = fn.stats
        hits = int(sum(stats.cache_hits.values()))
        misses = int(sum(stats.cache_misses.values()))
    except (AttributeError, TypeError):
        pass
    compile_times: list[float] = []
    asm_sizes: list[int] = []
    for sig in fn.signatures:
        try:
            timer = fn.get_metadata(sig).get("timers", {}).get("compiler_lock")
            if timer is not None:
                compile_times.append(float(timer))
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            # Numba returns an INVALID result (plus a UserWarning) when
            # asked to inspect code loaded from the on-disk cache — skip
            # the size in that case rather than report garbage.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                asm = fn.inspect_asm(sig)
            if not any("Inspection disabled" in str(w.message) for w in caught):
                asm_sizes.append(len(asm))
        except (AttributeError, KeyError, TypeError):
            pass
    # _cache is a private Numba attribute; degrade to False if it moves.
    cache = getattr(fn, "_cache", None)
    return FunctionReport(
        name=fn.py_func.__name__,
        signatures=tuple(str(sig) for sig in fn.signatures),
        cache_enabled=cache is not None
        and type(cache).__name__ != "NullCache",
        cache_hits=hits,
        cache_misses=misses,
        parallel=bool(opts.get("parallel", False)),
        fastmath=bool(opts.get("fastmath", False)),
        nogil=bool(opts.get("nogil", False)),
        boundscheck=bool(opts.get("boundscheck") or False),
        compile_times_s=tuple(compile_times),
        asm_text_bytes=tuple(asm_sizes),
    )


def _flag(value: bool) -> str:
    return "yes" if value else "no"


def show(fn: Any, *, verbose: bool = True) -> str:
    """Human-readable compilation report; printed unless ``verbose=False``,
    always returned as a string."""
    report = inspect(fn)
    lines = [f"Function: {report.name}", ""]
    lines.append("Compiled signatures:")
    if report.signatures:
        for sig in report.signatures:
            lines.append(f"  {sig}")
    else:
        lines.append("  (none — not compiled yet)")
    lines.append("")
    cache_state = "enabled" if report.cache_enabled else "disabled"
    lines.append(
        f"Cache: {cache_state} "
        f"(hits: {report.cache_hits}, misses: {report.cache_misses})"
    )
    lines.append(f"Parallel: {_flag(report.parallel)}")
    lines.append(f"Fastmath: {_flag(report.fastmath)}")
    lines.append(f"Nogil: {_flag(report.nogil)}")
    lines.append(f"Boundscheck: {_flag(report.boundscheck)}")
    if report.compile_times_s:
        formatted = ", ".join(f"{t * 1e3:.0f} ms" for t in report.compile_times_s)
        lines.append(f"Compilation time: {formatted}")
    if report.asm_text_bytes:
        formatted = ", ".join(f"{b / 1024:.1f} KB" for b in report.asm_text_bytes)
        lines.append(f"Assembly text size: {formatted}")
    text = "\n".join(lines)
    if verbose:
        print(text)
    return text


def check(fn: Any, *, verbose: bool = True) -> list[str]:
    """Warn about configurations with known failure modes.

    Returns the list of warnings (empty = nothing to flag); printed
    unless ``verbose=False``. These are heuristics distilled from
    production Numba workloads — each one links the relevant doc.
    """
    report = inspect(fn)
    warnings: list[str] = []
    if report.cache_enabled:
        warnings.append(
            "cache=True: on-disk cached binaries can crash intermittently "
            "when loaded by a process other than the one that compiled them "
            "(multi-process worker farms, network filesystems, ephemeral "
            "container storage). If you see intermittent access violations "
            "that disappear after deleting __pycache__, set "
            "NUMBA_UTILS_CACHE=0 in the environment before numba_utils "
            "is imported — configure(cache=False) reaches only functions "
            "decorated after the call, not numba-utils' own kernels. "
            "See docs/numba-cache.md."
        )
    if report.parallel:
        warnings.append(
            "parallel=True: every prange launch pays a thread-team barrier, "
            "so fine-grained parallel regions can be SLOWER than serial "
            "while still pinning all cores; repeated launches in one "
            "process have crashed threadpools on some setups. Prefer "
            "coarse, race-free parallelism, and consider process-level "
            "workers with NUMBA_NUM_THREADS=1. See docs/parallelism.md."
        )
    if report.fastmath:
        warnings.append(
            "fastmath=True: relaxes IEEE 754 (reassociation, no NaN/signed-"
            "zero guarantees). Results may differ across runs and machines; "
            "do not use where exact float semantics or reproducibility "
            "matter."
        )
    if report.boundscheck:
        warnings.append(
            "boundscheck=True: development mode — every array access is "
            "checked, which costs real throughput. Disable it in "
            "production builds."
        )
    if not report.signatures:
        warnings.append(
            "not compiled yet: the first call pays the full JIT cost. Call "
            "warmup() before timing or serving latency-sensitive traffic. "
            "See docs/benchmarking.md."
        )
    elif len(report.signatures) > _SIGNATURE_CHURN_THRESHOLD:
        warnings.append(
            f"{len(report.signatures)} compiled signatures: every new "
            "argument dtype combination triggers a full recompile. Check "
            "callers for unstable dtypes (int vs float literals, "
            "float32/float64 mixing)."
        )
    if verbose and warnings:
        print(f"diagnostics.check({report.name}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    return warnings
