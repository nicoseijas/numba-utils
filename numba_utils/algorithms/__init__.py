"""Selection and sorting algorithms specialized for Numba."""

from numba_utils.algorithms._combinatorics import combination_table
from numba_utils.algorithms._disjoint import (
    DisjointRankStructure,
    disjoint_rank_aggregate,
)
from numba_utils.algorithms._select import (
    fast_argpartition,
    nth_element,
    quickselect,
)
from numba_utils.algorithms._sort import (
    counting_sort,
    insertion_sort,
    lexsort,
    partial_sort,
    radix_sort,
    stable_argsort,
)
from numba_utils.algorithms._topk import argmax2, topk

__all__ = [
    "DisjointRankStructure",
    "argmax2",
    "combination_table",
    "counting_sort",
    "disjoint_rank_aggregate",
    "fast_argpartition",
    "insertion_sort",
    "lexsort",
    "nth_element",
    "partial_sort",
    "quickselect",
    "radix_sort",
    "stable_argsort",
    "topk",
]
