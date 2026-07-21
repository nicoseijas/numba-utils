"""ContentHashLocator: the stamp must see what (mtime, size) cannot."""

import os

import numpy as np

from numba_utils.cache_locator import ContentHashLocator


def _sum_impl(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total


class TestContentHashLocator:
    def test_detects_content_change_that_preserves_mtime_and_size(
        self, tmp_path
    ):
        # THE dangerous scenario from the 0.4.0 verdict: same size,
        # same mtime (docker COPY / tar -x / rsync -a / cp -p), new
        # content — Numba's default (mtime, size) stamp is identical,
        # so the binary compiled from the OLD source keeps loading.
        src = tmp_path / "kernel.py"
        src.write_bytes(b"X = 1  # version A\n")
        st = os.stat(src)

        locator = ContentHashLocator.__new__(ContentHashLocator)
        locator._py_file = str(src)
        stamp_a = locator.get_source_stamp()

        src.write_bytes(b"X = 2  # version B\n")  # same length
        os.utime(src, (st.st_atime, st.st_mtime))  # preserve mtime
        st2 = os.stat(src)
        assert (st.st_mtime, st.st_size) == (st2.st_mtime, st2.st_size)

        stamp_b = locator.get_source_stamp()
        assert stamp_a != stamp_b

    def test_stamp_is_stable_for_unchanged_content(self, tmp_path):
        src = tmp_path / "kernel.py"
        src.write_bytes(b"X = 1\n")
        locator = ContentHashLocator.__new__(ContentHashLocator)
        locator._py_file = str(src)
        assert locator.get_source_stamp() == locator.get_source_stamp()

    def test_numba_accepts_it_as_locator(self):
        # End to end IN ONE PROCESS (cross-process cache loads are the
        # segfault gotcha this library documents — not exercised here):
        # register via Numba's hook, compile a cached function, verify
        # the dispatcher's cache is driven by ContentHashLocator.
        from numba import njit
        from numba.core import config

        old = config.CACHE_LOCATOR_CLASSES
        config.CACHE_LOCATOR_CLASSES = (
            "numba_utils.cache_locator.ContentHashLocator"
        )
        try:
            fn = njit(cache=True)(_sum_impl)
            arr = np.arange(10, dtype=np.float64)
            assert fn(arr) == arr.sum()
            locator = fn._cache._impl._locator
            assert type(locator) is ContentHashLocator
            stamp = locator.get_source_stamp()
            assert stamp[0] == "sha256"
        finally:
            config.CACHE_LOCATOR_CLASSES = old
