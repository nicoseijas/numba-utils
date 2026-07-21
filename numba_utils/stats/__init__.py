"""Numerically hard statistics and transforms.

Deliberately small — the identity filter (ROADMAP.md) admits only
functions whose naive implementations are *wrong*, not just slow:

- :func:`logsumexp` / :func:`softmax` — the direct formulas overflow
  ``exp`` for inputs beyond ~709; these use the max-shift identity.
- :func:`weighted_quantile` — weighted quantiles have several
  incompatible conventions; this one implements inverted CDF, the one
  NumPy itself supports with weights, and validates its inputs.
"""

from numba_utils.stats._quantile import weighted_quantile
from numba_utils.stats._softmax import logsumexp, softmax

__all__ = [
    "logsumexp",
    "softmax",
    "weighted_quantile",
]
