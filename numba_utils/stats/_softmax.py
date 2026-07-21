"""Max-shifted exponential transforms: logsumexp and softmax."""

from __future__ import annotations

import numpy as np

from numba_utils.decorators import cached_njit


@cached_njit
def logsumexp(arr):
    """``log(sum(exp(arr)))`` of 1-D ``arr``, computed stably, as float64.

    The naive formula overflows ``exp`` for any element beyond ~709;
    this uses the max-shift identity ``m + log(sum(exp(arr - m)))``.
    An empty array returns ``-inf`` (the log of an empty sum), matching
    ``scipy.special.logsumexp``. A ``+inf`` element makes the result
    ``+inf``; NaN propagates.

    Complexity: O(n), two passes. Memory: O(1).
    """
    n = arr.shape[0]
    if n == 0:
        return -np.inf
    m = np.float64(arr[0])
    for i in range(1, n):
        x = np.float64(arr[i])
        if x > m:
            m = x
    if np.isnan(m):
        return np.nan
    if m == np.inf:
        return np.inf
    if m == -np.inf:
        # every element is -inf: sum of zeros
        return -np.inf
    acc = 0.0
    for i in range(n):
        acc += np.exp(np.float64(arr[i]) - m)
    return m + np.log(acc)


@cached_njit
def softmax(arr, out=None):
    """Softmax of 1-D ``arr`` as float64: ``exp(arr) / sum(exp(arr))``.

    Max-shifted for stability, so large inputs don't overflow ``exp``.
    Returns a new array, or writes into ``out`` (float64, same length)
    and returns it. Raises ``ValueError`` on empty input. Non-finite
    elements make the result undefined (NaN), as in the mathematical
    expression itself.

    Complexity: O(n), two passes. Memory: O(n), O(1) with ``out=``.
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("softmax: empty array")
    if out is None:
        out = np.empty(n, np.float64)
    if out.shape[0] != n:
        raise ValueError("softmax: out has wrong length")
    m = np.float64(arr[0])
    for i in range(1, n):
        x = np.float64(arr[i])
        if x > m:
            m = x
    acc = 0.0
    for i in range(n):
        e = np.exp(np.float64(arr[i]) - m)
        out[i] = e
        acc += e
    inv = 1.0 / acc
    for i in range(n):
        out[i] *= inv
    return out
