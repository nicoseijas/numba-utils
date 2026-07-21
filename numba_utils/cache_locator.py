"""Content-hash cache locator: close the stale-binary window.

Numba's default cache locator stamps a source file with
``(mtime, size)``. Deployment channels that PRESERVE mtime — ``docker
COPY`` (build-context files), ``tar -x``, ``rsync -a``, ``cp -p`` —
combined with a size collision make a new source file carry the old
stamp, so Numba silently keeps loading the binary compiled from the
PREVIOUS version of the code. A compile-time gate added in a release
never runs; nothing warns. (A plain ``pip install --upgrade`` is safe:
pip rewrites the ``.py`` files with a fresh mtime.)

:class:`ContentHashLocator` stamps by SHA-256 of the file contents
instead, which cannot collide across edits. Opt in via Numba's
official hook, BEFORE the first ``numba`` import::

    NUMBA_CACHE_LOCATOR_CLASSES=numba_utils.cache_locator.ContentHashLocator

Two caveats:

- Setting the variable REPLACES Numba's locator chain for every cached
  function in the process. ``ContentHashLocator`` covers the common
  case (a source file with a writable ``__pycache__``); to keep
  Numba's fallbacks for zip imports or unwritable trees, append their
  bare names: ``...ContentHashLocator,UserWideCacheLocator,ZipCacheLocator``.
- Cost: one file read + SHA-256 per cached function per process — noise
  next to a compile, but not zero.

This module deliberately imports nothing from ``numba_utils`` — Numba
imports it lazily while decorating the first cached function, which can
happen mid-import of the package itself.
"""

from __future__ import annotations

import hashlib

from numba.core.caching import InTreeCacheLocator


class ContentHashLocator(InTreeCacheLocator):
    """In-tree cache locator stamping by source-content hash.

    Identical to Numba's default in every way except
    :meth:`get_source_stamp`: the stamp is the SHA-256 of the source
    file's bytes, so a changed file ALWAYS invalidates the cached
    binary — even when a deployment preserved mtime and the size
    happens to match.
    """

    def get_source_stamp(self):
        with open(self._py_file, "rb") as f:
            return ("sha256", hashlib.sha256(f.read()).hexdigest())
