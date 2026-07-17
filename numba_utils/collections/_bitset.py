"""Fixed-capacity bit set."""

from __future__ import annotations

import numpy as np
from numba import int64, uint64
from numba.experimental import jitclass

_ONE = np.uint64(1)
_ZERO = np.uint64(0)
_WORD_MASK = np.uint64(63)


@jitclass([("_words", uint64[:]), ("_capacity", int64)])
class BitSet:
    """Set of ints in ``[0, capacity)`` packed 64 per word.

    O(1) add/discard/contains, O(capacity/64) count and clear, 8 bytes
    per 64 slots. Out-of-range indices raise ``IndexError``.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("BitSet: capacity must be >= 1")
        self._capacity = capacity
        self._words = np.zeros((capacity + 63) >> 6, np.uint64)

    def _check(self, i):
        if i < 0 or i >= self._capacity:
            raise IndexError("BitSet: index out of range")

    def add(self, i):
        self._check(i)
        self._words[i >> 6] |= _ONE << uint64(i & 63)

    def discard(self, i):
        self._check(i)
        self._words[i >> 6] &= ~(_ONE << uint64(i & 63))

    def contains(self, i):
        self._check(i)
        return (self._words[i >> 6] >> uint64(i & 63)) & _ONE != _ZERO

    def count(self):
        total = 0
        for w in self._words:
            while w != _ZERO:
                w &= w - _ONE
                total += 1
        return total

    def clear(self):
        for i in range(self._words.shape[0]):
            self._words[i] = _ZERO

    def capacity(self):
        return self._capacity
