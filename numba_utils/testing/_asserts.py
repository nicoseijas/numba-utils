"""Equivalence assertions between reference and jitted implementations."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np


def assert_close(
    actual: Any, desired: Any, *, rtol: float = 1e-9, atol: float = 0.0
) -> None:
    """Assert scalars/arrays are elementwise close (shape-checked).

    Tighter default (rtol 1e-9) than ``np.testing.assert_allclose``
    (1e-7): kernel refactors should preserve results to near machine
    precision unless documented otherwise (fastmath, parallel
    reassociation — loosen rtol explicitly there).
    """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def _compare_results(
    expected: Any, actual: Any, rtol: float, atol: float
) -> None:
    if isinstance(expected, tuple) or isinstance(actual, tuple):
        if not (isinstance(expected, tuple) and isinstance(actual, tuple)):
            raise AssertionError(
                f"result kinds differ: reference returned "
                f"{type(expected).__name__}, candidate {type(actual).__name__}"
            )
        if len(expected) != len(actual):
            raise AssertionError(
                f"tuple lengths differ: {len(expected)} vs {len(actual)}"
            )
        for j, (e, a) in enumerate(zip(expected, actual)):
            try:
                _compare_results(e, a, rtol, atol)
            except AssertionError as exc:
                raise AssertionError(f"tuple element {j}: {exc}") from exc
        return
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _copy_case(case: tuple) -> tuple:
    return tuple(
        np.copy(a) if isinstance(a, np.ndarray) else a for a in case
    )


def assert_equivalent(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    inputs: Iterable[Any],
    *,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> int:
    """Assert ``candidate`` matches ``reference`` on every input case.

    ``inputs`` yields argument tuples (a bare array counts as a 1-tuple)
    — combine with :func:`random_arrays` for generated cases. Each
    implementation receives its OWN copy of every array argument, so
    in-place kernels can't contaminate the comparison. Raises
    ``AssertionError`` naming the failing case index and input summary;
    returns the number of cases checked (guards against an accidentally
    empty generator passing vacuously).
    """
    if not callable(reference) or not callable(candidate):
        raise TypeError("assert_equivalent expects two callables")
    checked = 0
    for index, case in enumerate(inputs):
        if not isinstance(case, tuple):
            case = (case,)
        expected = reference(*_copy_case(case))
        actual = candidate(*_copy_case(case))
        try:
            _compare_results(expected, actual, rtol, atol)
        except AssertionError as exc:
            summary = ", ".join(
                f"ndarray(shape={a.shape}, dtype={a.dtype})"
                if isinstance(a, np.ndarray)
                else repr(a)
                for a in case
            )
            raise AssertionError(
                f"assert_equivalent: case {index} failed for inputs "
                f"({summary}): {exc}"
            ) from exc
        checked += 1
    if checked == 0:
        raise AssertionError("assert_equivalent: inputs yielded no cases")
    return checked
