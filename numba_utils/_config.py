"""Global configuration: code-level overrides with environment fallbacks.

Two levels, resolved per option at decoration time:

1. Code: ``configure(cache=False)`` or ``config.cache = False``.
2. Environment: ``NUMBA_UTILS_CACHE=0`` (checked only when the code level
   is unset).

A set option (``True``/``False``) is a GLOBAL override: it wins over
decorator defaults and per-call keyword arguments alike. That is the
point — overrides exist for environment-level policy, like disabling the
on-disk cache on machines where it is unsafe (see docs/numba-cache.md).
``None`` means "no override": decorator defaults and per-call arguments
apply unchanged.

Overrides affect only functions decorated AFTER the change — configure
before defining your jitted functions. numba-utils' own kernels are
decorated while ``numba_utils`` is imported, so code-level overrides can
never reach them: to cover the library's kernels too, set the
environment variable before the first import.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

OPTION_ENV_VARS = {
    "cache": "NUMBA_UTILS_CACHE",
    "fastmath": "NUMBA_UTILS_FASTMATH",
    "parallel": "NUMBA_UTILS_PARALLEL",
    "nogil": "NUMBA_UTILS_NOGIL",
}


class GlobalConfig:
    """Tri-state global overrides for every numba-utils decorator.

    Attributes ``cache``, ``fastmath``, ``parallel``, ``nogil`` are each
    ``None`` (no override), ``True`` or ``False`` (forced globally).
    """

    __slots__ = ("cache", "fastmath", "parallel", "nogil")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all code-level overrides. Environment fallbacks remain."""
        for name in OPTION_ENV_VARS:
            setattr(self, name, None)

    def resolve(self, name: str) -> bool | None:
        """Effective override for ``name``: code level, else environment,
        else ``None``. Unrecognized environment values count as unset."""
        value = getattr(self, name)
        if value is not None:
            return bool(value)
        raw = os.environ.get(OPTION_ENV_VARS[name], "").strip().lower()
        if raw in _TRUTHY:
            return True
        if raw in _FALSY:
            return False
        return None


config = GlobalConfig()


def configure(**options: bool | None) -> None:
    """Set global decorator overrides in code.

    ::

        import numba_utils as nu
        nu.configure(cache=False, fastmath=True)

    Valid options: ``cache``, ``fastmath``, ``parallel``, ``nogil``.
    Pass ``None`` to clear a single override; ``config.reset()`` clears
    all. Raises ``ValueError`` on unknown names (fail fast, no typos
    silently ignored).
    """
    for name, value in options.items():
        if name not in OPTION_ENV_VARS:
            valid = ", ".join(sorted(OPTION_ENV_VARS))
            raise ValueError(
                f"configure: unknown option {name!r}; valid options: {valid}"
            )
        if value is not None and not isinstance(value, bool):
            raise ValueError(
                f"configure: option {name!r} must be True, False or None"
            )
        setattr(config, name, value)
