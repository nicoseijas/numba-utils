"""JIT decorator aliases with sensible defaults."""

from numba_utils.decorators._wrappers import (
    CACHE_ENV_VAR,
    DEV_MODE_ENV_VAR,
    boundscheck,
    cached_njit,
    njit_fast,
    njit_parallel,
)

__all__ = [
    "CACHE_ENV_VAR",
    "DEV_MODE_ENV_VAR",
    "boundscheck",
    "cached_njit",
    "njit_fast",
    "njit_parallel",
]
