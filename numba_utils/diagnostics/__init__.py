"""Introspection and sanity checks for compiled dispatchers.

::

    from numba_utils import diagnostics

    diagnostics.show(foo)    # what did Numba actually build?
    diagnostics.check(foo)   # known-issue warnings with recommendations
    diagnostics.shadowed()   # is any module loaded from a stale file?
"""

from numba_utils.diagnostics._inspect import FunctionReport, check, inspect, show
from numba_utils.diagnostics._shadowing import ShadowedModule, shadowed

__all__ = [
    "FunctionReport",
    "ShadowedModule",
    "check",
    "inspect",
    "shadowed",
    "show",
]
