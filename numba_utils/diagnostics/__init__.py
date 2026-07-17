"""Introspection and sanity checks for compiled dispatchers.

::

    from numba_utils import diagnostics

    diagnostics.show(foo)    # what did Numba actually build?
    diagnostics.check(foo)   # known-issue warnings with recommendations
"""

from numba_utils.diagnostics._inspect import FunctionReport, check, inspect, show

__all__ = ["FunctionReport", "check", "inspect", "show"]
