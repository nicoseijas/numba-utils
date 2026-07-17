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
    if func is None:
        return lambda f: _apply_njit(f, options, {})
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
    disappear after deleting ``__pycache__``, disable caching globally
    with ``configure(cache=False)`` or ``NUMBA_UTILS_CACHE=0`` — either
    overrides even explicit ``cache=True``. Full story:
    docs/numba-cache.md.
    """
    return _apply_njit(func, {"cache": True}, overrides)


def boundscheck(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """Development-mode bounds checking that vanishes in production.

    With ``NUMBA_UTILS_DEV=1`` in the environment, compiles with
    ``boundscheck=True`` so out-of-bounds array access raises ``IndexError``
    (cache disabled: dev builds must not poison the on-disk cache).
    Without it, compiles a plain cached ``njit`` with zero overhead.

    The environment is read at decoration time, not call time.
    """
    if _dev_mode_enabled():
        defaults: dict[str, Any] = {"boundscheck": True, "cache": False}
    else:
        defaults = {"cache": True}
    return _apply_njit(func, defaults, overrides)
