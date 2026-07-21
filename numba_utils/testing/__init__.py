"""Helpers for testing jitted kernels with confidence.

The core workflow — validate every njit kernel against an independent
reference (nopython mode has no bounds checking; sizing bugs corrupt
memory silently instead of crashing)::

    from numba_utils.testing import assert_equivalent, random_arrays

    assert_equivalent(
        python_impl,
        njit_impl,
        random_arrays(n_cases=20, size=1000),
    )
"""

from numba_utils.testing._asserts import assert_close, assert_equivalent
from numba_utils.testing._generators import deterministic_rng, random_arrays
from numba_utils.testing._stochastic import (
    assert_converges,
    assert_reproducible,
)

__all__ = [
    "assert_close",
    "assert_converges",
    "assert_equivalent",
    "assert_reproducible",
    "deterministic_rng",
    "random_arrays",
]
