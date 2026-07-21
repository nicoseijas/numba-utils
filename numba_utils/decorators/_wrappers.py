"""Thin aliases over ``numba.njit`` with curated defaults.

Every decorator here accepts both bare and called forms::

    @njit_fast
    def foo(): ...

    @njit_fast(fastmath=False)
    def bar(): ...

Keyword arguments override the defaults and are forwarded to ``njit``
verbatim, so nothing about Numba is hidden — these are conveniences,
not abstractions.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable

from numba import njit

from numba_utils._config import OPTION_ENV_VARS, config

DEV_MODE_ENV_VAR = "NUMBA_UTILS_DEV"
CACHE_ENV_VAR = OPTION_ENV_VARS["cache"]

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _dev_mode_enabled() -> bool:
    return os.environ.get(DEV_MODE_ENV_VAR, "").strip().lower() in _TRUTHY


def _apply_njit(
    func: Callable[..., Any] | None,
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    locked: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    options = {**defaults, **overrides}
    # Global overrides (configure()/NUMBA_UTILS_* env) win over per-call
    # arguments by design: they exist for environment-level policy, e.g.
    # disabling the on-disk cache on machines where a binary loaded by a
    # process other than the compiling one segfaults intermittently
    # (0xC0000005 on Windows multi-process farms; docs/numba-cache.md).
    for name in OPTION_ENV_VARS:
        forced = config.resolve(name)
        if forced is not None:
            options[name] = forced
    # `locked` wins over EVERYTHING, including global overrides — used
    # where an option is a safety invariant, not a preference (dev-mode
    # boundscheck must never touch the on-disk cache).
    if locked:
        options.update(locked)
    # Library-wide invariant, enforced at the layer where ALL option
    # paths converge (decorator defaults, per-call kwargs, global
    # overrides): the on-disk cache must never see a boundscheck
    # build. Numba's cache key ignores boundscheck, so one cached
    # binary would poison checked and unchecked callers alike —
    # including cached_njit(boundscheck=True), which bypasses the
    # boundscheck() decorator entirely.
    if options.get("boundscheck"):
        # Warn if this silently overrides a DELIBERATE cache=True —
        # whether a per-call kwarg or the global override
        # (configure()/NUMBA_UTILS_CACHE, which wins over kwargs, so it
        # is what actually got overridden when set) — matching
        # configure()'s fail-fast stance. The override stands (it is a
        # safety invariant) but is audible. Decorator DEFAULTS stay
        # silent: nobody asked for them by name.
        forced_cache = config.resolve("cache")
        if forced_cache is True:
            cache_source = (
                "the global cache=True override (configure()/"
                f"{CACHE_ENV_VAR})"
            )
        elif forced_cache is None and overrides.get("cache") is True:
            cache_source = "your explicit cache=True"
        else:
            cache_source = None
        if cache_source is not None:
            warnings.warn(
                "boundscheck=True forces cache=False (Numba's cache key "
                "ignores boundscheck, so a shared cache would poison "
                f"checked and unchecked callers) — {cache_source} was "
                "overridden.",
                RuntimeWarning,
                stacklevel=3,
            )
        options["cache"] = False
    if func is None:
        return lambda f: _apply_njit(f, options, {}, locked)
    if not callable(func):
        raise TypeError(
            f"expected a callable to decorate, got {type(func).__name__!r}"
        )
    return njit(**options)(func)


def njit_fast(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """``njit`` tuned for throughput: ``cache=True, fastmath=True, nogil=True``.

    ``fastmath=True`` relaxes IEEE 754 semantics (reassociation, no signed
    zeros/NaN guarantees); don't use it where exact float semantics matter.
    Integer-only kernels (bitmasks, table lookups) gain nothing from
    ``fastmath`` — plain :func:`cached_njit` is the right pick there.
    """
    return _apply_njit(
        func, {"cache": True, "fastmath": True, "nogil": True}, overrides
    )


def njit_parallel(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """Clean alias for ``njit(parallel=True, cache=True)``.

    Before reaching for this, read docs/parallelism.md: fine-grained
    ``prange`` regions lose to serial code. For complete parallel
    operations with the pitfalls already engineered around, see
    ``numba_utils.parallel``.
    """
    return _apply_njit(func, {"parallel": True, "cache": True}, overrides)


def cached_njit(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """``njit(cache=True)``: compile once, reuse across runs. Ideal for scripts.

    Caveat: Numba's on-disk cache assumes the cached binary is safe to
    load from any process. On some setups (observed on Windows machines
    running multi-process farms) loading a binary compiled by another
    process crashes intermittently. If you see "random" segfaults that
    disappear after deleting ``__pycache__``, set ``NUMBA_UTILS_CACHE=0``
    before importing numba_utils — it overrides even explicit
    ``cache=True``, for your kernels and the library's own alike.
    (``configure(cache=False)`` covers only functions decorated after
    the call.) Full story: docs/numba-cache.md.
    """
    return _apply_njit(func, {"cache": True}, overrides)


def boundscheck(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """Development-mode bounds checking that vanishes in production.

    With ``NUMBA_UTILS_DEV=1`` in the environment, compiles with
    ``boundscheck=True`` so out-of-bounds array access raises
    ``IndexError``. In dev mode the on-disk cache is HARD-disabled — no
    global override or per-call argument can re-enable it. That is a
    safety invariant, not a preference: Numba's cache key does not
    include ``boundscheck``, so a shared cache poisons both directions
    — production loading a checked binary pays the cost, and (worse)
    dev loading an unchecked binary silently checks nothing while you
    believe it does. Without ``NUMBA_UTILS_DEV``, compiles a plain
    cached ``njit`` with zero overhead.

    The environment is read at decoration time, not call time.
    """
    if _dev_mode_enabled():
        return _apply_njit(
            func, {"boundscheck": True}, overrides, locked={"cache": False}
        )
    return _apply_njit(func, {"cache": True}, overrides)
