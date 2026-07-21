"""Shared value-type validation for dtype-generic container factories."""

from __future__ import annotations

import numpy as np
from numba.core import types as nb_types

# The supported Numba-type -> NumPy-dtype conversion; jitclass method
# bodies allocate their backing arrays with the NumPy dtype.
from numba.np.numpy_support import as_dtype


def validate_value_type(
    factory_name: str, value_type, *, ordered: bool = False
) -> np.dtype:
    """Return the NumPy dtype for a Numba scalar ``value_type``.

    Raises ``TypeError`` for anything that is not a Numba scalar number
    or boolean type and — when ``ordered`` — for complex types, which
    lack the comparisons an ordered container needs.
    """
    if not isinstance(value_type, (nb_types.Number, nb_types.Boolean)):
        raise TypeError(
            f"{factory_name} expects a numba scalar type "
            f"(e.g. int64, float32), got {value_type!r}"
        )
    if ordered and isinstance(value_type, nb_types.Complex):
        raise TypeError(
            f"{factory_name}: complex values have no ordering"
        )
    return np.dtype(as_dtype(value_type))
