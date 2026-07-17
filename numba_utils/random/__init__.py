"""Sampling utilities over Numba's internal RNG.

All functions here use Numba's nopython random state, which is SEPARATE
from NumPy's Python-level RNG: seed it with :func:`seed`, not
``np.random.seed`` called from Python.
"""

from numba_utils.random._rng import choice, permutation, seed, shuffle
from numba_utils.random._sampling import (
    alias_draw,
    alias_sample,
    alias_setup,
    reservoir_sampling,
    weighted_sampling,
)

__all__ = [
    "alias_draw",
    "alias_sample",
    "alias_setup",
    "choice",
    "permutation",
    "reservoir_sampling",
    "seed",
    "shuffle",
    "weighted_sampling",
]
