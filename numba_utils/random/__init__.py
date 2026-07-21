"""Sampling utilities over Numba's internal RNG, plus a stateless
counter-based generator.

The ``seed``/``shuffle``/sampling functions use Numba's nopython
random state, which is SEPARATE from NumPy's Python-level RNG: seed it
with :func:`seed`, not ``np.random.seed`` called from Python — and
note it is per-thread, so results can depend on thread scheduling.
The ``philox_*`` functions are the alternative without state at all:
pure functions of ``(key, counter)``, reproducible regardless of
threads or call order.
"""

from numba_utils.random._philox import (
    philox4x64,
    philox_randint,
    philox_uniform,
    philox_uniforms,
)
from numba_utils.random._rng import choice, permutation, seed, shuffle
from numba_utils.random._sampling import (
    alias_draw,
    alias_sample,
    alias_setup,
    partial_shuffle,
    reservoir_sampling,
    sample_without_replacement,
    weighted_sampling,
)

__all__ = [
    "alias_draw",
    "alias_sample",
    "alias_setup",
    "choice",
    "partial_shuffle",
    "permutation",
    "philox4x64",
    "philox_randint",
    "philox_uniform",
    "philox_uniforms",
    "reservoir_sampling",
    "sample_without_replacement",
    "seed",
    "shuffle",
    "weighted_sampling",
]
