"""typed.Dict conveniences: constructor sugar and counting."""

from __future__ import annotations

from numba import typed
from numba.core import types as nb_types

from numba_utils.decorators import cached_njit


def typed_defaultdict(key_type, value_type):
    """Empty ``numba.typed.Dict`` with declared key/value types.

    ::

        from numba import int64, float64
        d = typed_defaultdict(key_type=int64, value_type=float64)

    Pass the result into jitted functions; inside them, read with
    ``d.get(key, default)`` for defaultdict-style access. (Inside
    ``@njit`` code you can simply write ``d = dict()`` — this helper is
    for building typed dicts at the Python boundary, where the types
    must be declared.)
    """
    if not isinstance(key_type, nb_types.Type) or not isinstance(
        value_type, nb_types.Type
    ):
        raise TypeError(
            "typed_defaultdict expects numba types, e.g. "
            "typed_defaultdict(int64, float64)"
        )
    return typed.Dict.empty(key_type, value_type)


@cached_njit
def counter(arr):
    """Occurrence counts of 1-D ``arr`` as a typed dict ``{value: count}``.

    Works for any hashable element dtype (floats included). Honest
    positioning: for one-shot counting of a materialized array,
    ``np.unique(return_counts=True)`` (sort-based) and
    :func:`numba_utils.bincount` (dense ints) are FASTER — see
    BENCHMARKS.md. ``counter``'s value is incremental counting inside
    jitted loops, where materializing an array first is the expensive
    part.

    Complexity: O(n) expected. Memory: O(u) for u distinct values.
    """
    counts = dict()
    for i in range(arr.shape[0]):
        key = arr[i]
        # Membership + assignment (not .get with default): the only
        # pattern Numba's type inference resolves on a fresh dict().
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
    return counts
