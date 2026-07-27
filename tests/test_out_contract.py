"""The ``out=`` dtype contract, pinned across every module that has one.

A wrong-dtype ``out=`` buffer fails LOUDLY at compile time: the
``out is None`` branch allocates a specific dtype, and the argument must
unify with it, so Numba raises ``TypingError`` instead of truncating.
That loudness was an accident of the internal allocation until these
tests existed — a refactor that removed or retyped the allocation branch
would have silently turned it into truncation (audit issue #8).
"""

from __future__ import annotations

import numpy as np
import pytest
from numba.core.errors import TypingError

from numba_utils import (
    cumulative_sum,
    fast_clip,
    normalize,
    parallel_prefix_sum,
    rolling_mean,
    rolling_sum,
    softmax,
)
from numba_utils.random import philox_uniforms


def _buf(dtype, n=3):
    return np.empty(n, dtype)


# (id, call with a wrong-dtype out=). Two directions per function where
# both are meaningful: a narrower float, and int-vs-float in whichever
# direction the function's own allocation makes wrong.
WRONG_DTYPE_CALLS = [
    ("fast_clip/float32", lambda: fast_clip(np.ones(3), 0.0, 1.0, _buf(np.float32))),
    ("fast_clip/int64", lambda: fast_clip(np.ones(3), 0.0, 1.0, _buf(np.int64))),
    ("normalize/float32", lambda: normalize(np.ones(3), _buf(np.float32))),
    ("normalize/int64", lambda: normalize(np.ones(3), _buf(np.int64))),
    ("cumulative_sum/float32", lambda: cumulative_sum(np.ones(3), _buf(np.float32))),
    (
        "cumulative_sum/float64-for-int-input",
        lambda: cumulative_sum(np.ones(3, np.int64), _buf(np.float64)),
    ),
    ("rolling_sum/float32", lambda: rolling_sum(np.ones(5), 2, _buf(np.float32, 4))),
    ("rolling_sum/int64", lambda: rolling_sum(np.ones(5), 2, _buf(np.int64, 4))),
    ("rolling_mean/float32", lambda: rolling_mean(np.ones(5), 2, _buf(np.float32, 4))),
    (
        "parallel_prefix_sum/float32",
        lambda: parallel_prefix_sum(np.ones(5), _buf(np.float32, 5)),
    ),
    (
        "parallel_prefix_sum/int64",
        lambda: parallel_prefix_sum(np.ones(5), _buf(np.int64, 5)),
    ),
    ("softmax/float32", lambda: softmax(np.ones(3), _buf(np.float32))),
    ("softmax/int64", lambda: softmax(np.ones(3), _buf(np.int64))),
    ("philox_uniforms/float32", lambda: philox_uniforms(1, 0, 4, _buf(np.float32, 4))),
]

# Non-1-D buffers fail through the same unification; one representative
# per allocation style (empty_like vs explicit float64).
WRONG_NDIM_CALLS = [
    ("cumulative_sum/2d", lambda: cumulative_sum(np.ones(3), np.empty((3, 1)))),
    ("normalize/2d", lambda: normalize(np.ones(3), np.empty((3, 1)))),
]


@pytest.mark.parametrize(
    "call",
    [c for _, c in WRONG_DTYPE_CALLS],
    ids=[i for i, _ in WRONG_DTYPE_CALLS],
)
def test_wrong_dtype_out_raises_typing_error(call):
    with pytest.raises(TypingError):
        call()


@pytest.mark.parametrize(
    "call",
    [c for _, c in WRONG_NDIM_CALLS],
    ids=[i for i, _ in WRONG_NDIM_CALLS],
)
def test_non_1d_out_raises_typing_error(call):
    with pytest.raises(TypingError):
        call()


class TestRightDtypeStillWorks:
    """The mirror of the above: the contract is a dtype match, not a ban
    on ``out=``. Without these, an implementation that rejected EVERY
    buffer would pass the tests above.
    """

    def test_matching_buffers_are_written_in_place(self):
        arr = np.array([3.0, 1.0, 2.0])

        clipped = _buf(np.float64)
        assert fast_clip(arr, 1.5, 2.5, clipped) is clipped

        scaled = _buf(np.float64)
        assert normalize(arr, scaled) is scaled

        cumsum = _buf(np.float64)
        assert cumulative_sum(arr, cumsum) is cumsum

        rolled = _buf(np.float64, 2)
        assert rolling_sum(arr, 2, rolled) is rolled

        prefix = _buf(np.float64)
        assert parallel_prefix_sum(arr, prefix) is prefix

        probs = _buf(np.float64)
        assert softmax(arr, probs) is probs

        draws = _buf(np.float64, 4)
        assert philox_uniforms(1, 0, 4, draws) is draws

    def test_int_input_takes_an_int_buffer(self):
        # cumulative_sum and fast_clip allocate with empty_like, so the
        # matching buffer for an int array is an int buffer
        arr = np.array([1, 2, 3], np.int64)
        np.testing.assert_array_equal(
            cumulative_sum(arr, _buf(np.int64)), [1, 3, 6]
        )
        np.testing.assert_array_equal(
            fast_clip(arr, 1, 2, _buf(np.int64)), [1, 2, 2]
        )
