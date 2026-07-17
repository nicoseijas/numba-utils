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

DEV_MODE_ENV_VAR = "NUMBA_UTILS_DEV"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _dev_mode_enabled() -> bool:
    return os.environ.get(DEV_MODE_ENV_VAR, "").strip().lower() in _TRUTHY


def _apply_njit(
    func: Callable[..., Any] | None,
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> Callable[..., Any]:
    options = {**defaults, **overrides}
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


def parallel(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """Clean alias for ``njit(parallel=True, cache=True)``."""
    return _apply_njit(func, {"parallel": True, "cache": True}, overrides)


def cached_njit(
    func: Callable[..., Any] | None = None, /, **overrides: Any
) -> Callable[..., Any]:
    """``njit(cache=True)``: compile once, reuse across runs. Ideal for scripts."""
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
