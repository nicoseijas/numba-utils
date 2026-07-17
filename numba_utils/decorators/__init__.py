"""JIT decorator aliases with sensible defaults."""

from numba_utils.decorators._wrappers import (
    DEV_MODE_ENV_VAR,
    boundscheck,
    cached_njit,
    njit_fast,
    parallel,
)

__all__ = [
    "DEV_MODE_ENV_VAR",
    "boundscheck",
    "cached_njit",
    "njit_fast",
    "parallel",
]
