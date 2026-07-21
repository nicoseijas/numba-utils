"""Reach-weighted all-pairs comparison with exact set-disjointness.

The problem shape: N "hero" items and M "opponent" items, each
carrying K distinct integer keys and a sortable score; opponents carry
weights. Per hero ``i`` compute::

    win[i]  = sum_j (score_i > score_j) * disjoint(i, j) * w[j]
    lose[i] = sum_j (score_i < score_j) * disjoint(i, j) * w[j]
    tie[i]  = sum_j (score_i == score_j) * disjoint(i, j) * w[j]

where ``disjoint(i, j)`` means the two key sets share no element.
Dense is O(N·M). This computes it EXACTLY in O((2^K − 1) · (N + M)
log(N + M)) via inclusion–exclusion: "opponents with lower score"
minus "opponents with lower score sharing >= 1 key", the second term
expanded over the hero's 2^K − 1 nonempty key subsets with IE signs —
each key-clashing opponent is subtracted exactly once, an algebraic
identity, not tuning.

Contributed from a production PLO5 CFR solver, where it replaced the
dense showdown bottleneck; certified there against the dense reference
with a drop-removal mutation that screams (the same certification
ships in this library's tests, via ``testing.mutation_screams``).

Driven from Python (the build is NumPy lexsort work; the eval kernels
are jitted) — unlike most of the library these entry points are NOT
callable inside ``@njit``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numba import prange

from numba_utils.decorators import cached_njit, njit_parallel

_PARALLEL_THRESHOLD = 1 << 16
_MAX_K = 12
_MAX_CODE = 1 << 62
# Cap on the combinadic binomial table, in int64 entries (2**24 =
# 128 MiB). Only reachable for K <= 3, where the positional encoding
# already covers V in the hundreds of millions.
_MAX_COMB_TABLE = 1 << 24


# cache=False on the parallel kernels, per house convention for
# parfor-transformed functions.
@njit_parallel(cache=False)
def _seg_at_q(val, order, gstarts, gends, qid_row, out, total_rows):
    # Per-group running sum, recorded at each query row. Groups are
    # independent; serial below the launch-barrier threshold.
    if total_rows < _PARALLEL_THRESHOLD:
        for g in range(gstarts.shape[0]):
            run = 0.0
            for r in range(gstarts[g], gends[g]):
                run += val[order[r]]
                q = qid_row[r]
                if q >= 0:
                    out[q] = run
    else:
        for g in prange(gstarts.shape[0]):
            run = 0.0
            for r in range(gstarts[g], gends[g]):
                run += val[order[r]]
                q = qid_row[r]
                if q >= 0:
                    out[q] = run


@njit_parallel(cache=False)
def _group_tot(val, order, gstarts, gends, out_g, total_rows):
    if total_rows < _PARALLEL_THRESHOLD:
        for g in range(gstarts.shape[0]):
            s = 0.0
            for r in range(gstarts[g], gends[g]):
                s += val[order[r]]
            out_g[g] = s
    else:
        for g in prange(gstarts.shape[0]):
            s = 0.0
            for r in range(gstarts[g], gends[g]):
                s += val[order[r]]
            out_g[g] = s


@njit_parallel(cache=False)
def _accum(le, lt, gtot, q_group, s_eff, n_hero, n_rows_per_hero, win, lose, tie):
    # Per-hero accumulation (race-free: each hero owns its rows).
    # Hero i's query rows sit at q = i + n_hero * k.
    if n_hero < _PARALLEL_THRESHOLD:
        for i in range(n_hero):
            w = 0.0
            l = 0.0
            t = 0.0
            for k in range(n_rows_per_hero):
                q = i + n_hero * k
                s = s_eff[q]
                leq = le[q]
                ltq = lt[q]
                totq = gtot[q_group[q]]
                w += s * ltq
                l += s * (totq - leq)
                t += s * (leq - ltq)
            win[i] = w
            lose[i] = l
            tie[i] = t
    else:
        for i in prange(n_hero):
            w = 0.0
            l = 0.0
            t = 0.0
            for k in range(n_rows_per_hero):
                q = i + n_hero * k
                s = s_eff[q]
                leq = le[q]
                ltq = lt[q]
                totq = gtot[q_group[q]]
                w += s * ltq
                l += s * (totq - leq)
                t += s * (leq - ltq)
            win[i] = w
            lose[i] = l
            tie[i] = t


def _validate_keys_scores(name, keys, scores):
    keys = np.asarray(keys)
    scores = np.asarray(scores)
    if keys.ndim != 2:
        raise ValueError(f"disjoint_rank_aggregate: {name}_keys must be 2-D (N, K)")
    if keys.shape[0] < 1:
        raise ValueError(f"disjoint_rank_aggregate: {name} side is empty")
    if scores.shape != (keys.shape[0],):
        raise ValueError(
            f"disjoint_rank_aggregate: {name}_scores length must match {name}_keys rows"
        )
    if not np.issubdtype(keys.dtype, np.integer):
        raise ValueError(f"disjoint_rank_aggregate: {name}_keys must be integers")
    if np.issubdtype(scores.dtype, np.floating) and not np.all(np.isfinite(scores)):
        raise ValueError(f"disjoint_rank_aggregate: {name}_scores must be finite")
    srt = np.sort(keys, axis=1)
    if np.any(srt[:, 1:] == srt[:, :-1]):
        raise ValueError(
            f"disjoint_rank_aggregate: {name}_keys rows must contain "
            "DISTINCT keys — inclusion-exclusion over subsets assumes a "
            "set, and a duplicated key breaks the identity"
        )
    return keys.astype(np.int64), scores


def _subset_combos(k):
    combos = []
    for s in range(1, k + 1):
        sign = 1.0 if (s % 2 == 1) else -1.0
        for c in itertools.combinations(range(k), s):
            combos.append((c, sign))
    return combos


def _encode(keys_ranked, positions, base, k_total):
    sub = np.sort(keys_ranked[:, list(positions)], axis=1)
    code = np.zeros(sub.shape[0], np.int64)
    for j in range(sub.shape[1]):
        code = code * base + (sub[:, j] + 1)
    # size in the low bits so different-size subsets never collide
    return code * (k_total + 1) + sub.shape[1]


def _binom_table(v_count, k):
    # comb[v, j] = C(v, j) for 0 <= v < v_count, 0 <= j <= k. Built by
    # the hockey-stick identity C(v, j) = Σ_{u<v} C(u, j-1); every
    # entry is <= max_s C(v_count, s), which the caller has already
    # guarded below 2**62 / (k + 1), so int64 never overflows here.
    comb = np.zeros((v_count, k + 1), np.int64)
    comb[:, 0] = 1
    for j in range(1, k + 1):
        comb[1:, j] = np.cumsum(comb[:-1, j - 1])
    return comb


def _encode_combinadic(keys_ranked, positions, comb, k_total):
    # Combinatorial number system: a sorted subset {v_1 < ... < v_s}
    # of ranks 0..V-1 maps bijectively to Σ_j C(v_j, j), a rank in
    # [0, C(V, s)) — position-independent, so hero and opponent
    # subsets with the same values land in the same group.
    sub = np.sort(keys_ranked[:, list(positions)], axis=1)
    rank = np.zeros(sub.shape[0], np.int64)
    for j in range(sub.shape[1]):
        rank += comb[sub[:, j], j + 1]
    return rank * (k_total + 1) + sub.shape[1]


def _combinadic_limit(v_count, k):
    return max(math.comb(v_count, s) for s in range(1, k + 1)) * (k + 1)


def _domain_cap(k):
    """Largest number of distinct key values accepted at this K
    (either encoding). Exact-integer binary search; the accepted set
    is a union of two prefixes, hence a prefix."""

    def ok(v):
        if ((v + 1) ** k) * (k + 1) < _MAX_CODE:
            return True
        return (
            _combinadic_limit(v, k) < _MAX_CODE
            and v * (k + 1) <= _MAX_COMB_TABLE
        )

    lo, hi = 1, 1 << 62
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


class DisjointRankStructure:
    """Prebuilt incidence structure: ``build`` once, ``eval`` per
    weight vector.

    The sort order and group topology depend only on keys and scores —
    in iterative workloads (CFR: regrets change, topology doesn't) the
    O((2^K − 1)·N log N) sort work is paid once and each ``eval`` is a
    few segmented-cumsum passes. For a single weight vector use
    :func:`disjoint_rank_aggregate`.
    """

    __slots__ = (
        "n_hero", "n_opp", "_n_rows_per_hero", "_nR", "_nQ", "_rj",
        "_s_eff", "_ngroups", "_q_group", "_valbuf",
        "_order_le", "_gs_le", "_ge_le", "_qidrow_le",
        "_order_lt", "_gs_lt", "_ge_lt", "_qidrow_lt",
    )

    @classmethod
    def build(cls, hero_keys, hero_scores, opp_keys, opp_scores):
        hero_keys, hero_scores = _validate_keys_scores("hero", hero_keys, hero_scores)
        opp_keys, opp_scores = _validate_keys_scores("opp", opp_keys, opp_scores)
        if hero_keys.shape[1] != opp_keys.shape[1]:
            raise ValueError(
                "disjoint_rank_aggregate: hero and opp must have the same K"
            )
        k = hero_keys.shape[1]
        if k > _MAX_K:
            raise ValueError(
                f"disjoint_rank_aggregate: K={k} means 2^{k}-1 subsets per "
                f"item — beyond K={_MAX_K} the expansion dominates; this "
                "algorithm targets small key sets"
            )
        # Rank-compress the key domain so ANY int64 values work. Subset
        # codes must fit in int64; two encodings, widest that fits:
        # - positional (digits base V+1): free, fits while
        #   (V+1)^K·(K+1) < 2**62;
        # - combinadic (combinatorial number system): the subset's rank
        #   among C(V, s) — a K!-wider envelope, C(V,K)·(K+1) < 2**62,
        #   at the cost of a (V, K+1) binomial table. This is what lets
        #   a 52-card deck through at every K <= 12 (positional capped
        #   it at K <= 10 / 28 distinct values at K=12).
        uniq, inv = np.unique(
            np.concatenate([hero_keys.ravel(), opp_keys.ravel()]),
            return_inverse=True,
        )
        v_count = len(uniq)
        base = v_count + 1
        if (base**k) * (k + 1) < _MAX_CODE:
            def _enc(mat, pos):
                return _encode(mat, pos, base, k)
        elif (
            _combinadic_limit(v_count, k) < _MAX_CODE
            and v_count * (k + 1) <= _MAX_COMB_TABLE
        ):
            binom = _binom_table(v_count, k)

            def _enc(mat, pos):
                return _encode_combinadic(mat, pos, binom, k)
        else:
            raise ValueError(
                f"disjoint_rank_aggregate: {v_count} distinct key values "
                f"with K={k} overflow the subset encoding (the cap at "
                f"K={k} is {_domain_cap(k)} distinct values) — reduce "
                "the key domain or K"
            )
        n_hero_total = hero_keys.shape[0] * k
        hk = inv[:n_hero_total].reshape(hero_keys.shape).astype(np.int64)
        ok = inv[n_hero_total:].reshape(opp_keys.shape).astype(np.int64)

        st = cls()
        st.n_hero = hero_keys.shape[0]
        st.n_opp = opp_keys.shape[0]
        combos = _subset_combos(k)
        st._n_rows_per_hero = len(combos) + 1

        # Opponent rows: one global group (key 0) + one row per subset.
        rk = [np.zeros(st.n_opp, np.int64)]
        rj = [np.arange(st.n_opp)]
        rs = [opp_scores]
        for pos, _sign in combos:
            rk.append(_enc(ok, pos))
            rj.append(np.arange(st.n_opp))
            rs.append(opp_scores)
        rk = np.concatenate(rk)
        st._rj = np.concatenate(rj)
        rs = np.concatenate(rs)
        st._nR = len(rk)

        # Hero query rows: global (+1) + subsets with -IE sign
        # (win = unblocked - blocked).
        qk = [np.zeros(st.n_hero, np.int64)]
        qsc = [hero_scores]
        seff = [np.ones(st.n_hero)]
        for pos, sign in combos:
            qk.append(_enc(hk, pos))
            qsc.append(hero_scores)
            seff.append(np.full(st.n_hero, -sign))
        qk = np.concatenate(qk)
        qsc = np.concatenate(qsc)
        st._s_eff = np.concatenate(seff)
        st._nQ = len(qk)

        key = np.concatenate([rk, qk])
        score = np.concatenate([rs, qsc])
        is_q = np.concatenate(
            [np.zeros(st._nR, bool), np.ones(st._nQ, bool)]
        )
        qidx = np.concatenate(
            [np.full(st._nR, -1, np.int64), np.arange(st._nQ, dtype=np.int64)]
        )
        # LE order: opp rows before queries at (key, score) ties -> the
        # running sum at a query includes equal-score opponents.
        # LT order: queries first -> strictly-lower only.
        for name, tag in (("le", is_q.astype(np.int64)), ("lt", (~is_q).astype(np.int64))):
            order = np.lexsort((tag, score, key))
            sk = key[order]
            gflag = np.empty(len(sk), np.bool_)
            gflag[0] = True
            gflag[1:] = sk[1:] != sk[:-1]
            gstarts = np.where(gflag)[0].astype(np.int64)
            gends = np.append(gstarts[1:], len(sk)).astype(np.int64)
            qidrow = qidx[order]
            setattr(st, f"_order_{name}", order)
            setattr(st, f"_gs_{name}", gstarts)
            setattr(st, f"_ge_{name}", gends)
            setattr(st, f"_qidrow_{name}", qidrow)
            if name == "le":
                gid_row = np.cumsum(gflag) - 1
                st._ngroups = int(gid_row[-1]) + 1
                qmask = is_q[order]
                st._q_group = np.empty(st._nQ, np.int64)
                st._q_group[qidx[order][qmask]] = gid_row[qmask]
        st._valbuf = np.zeros(st._nR + st._nQ)
        return st

    def eval(self, opp_w):
        """``(win, lose, tie)`` for one opponent weight vector.

        O((2^K − 1)·(N + M)) — no sorting; the topology is prebuilt.
        Weights must be finite and non-negative (NaN would corrupt the
        prefix sums silently — validated up front, the
        `weighted_sampling` lesson).
        """
        w = np.asarray(opp_w, np.float64)
        if w.shape != (self.n_opp,):
            raise ValueError(
                "disjoint_rank_aggregate: weights length must match opp side"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(
                "disjoint_rank_aggregate: weights must be finite and >= 0"
            )
        total_rows = self._nR + self._nQ
        val = self._valbuf
        val[: self._nR] = w[self._rj]  # query part stays 0
        le = np.zeros(self._nQ)
        lt = np.zeros(self._nQ)
        _seg_at_q(val, self._order_le, self._gs_le, self._ge_le,
                  self._qidrow_le, le, total_rows)
        _seg_at_q(val, self._order_lt, self._gs_lt, self._ge_lt,
                  self._qidrow_lt, lt, total_rows)
        gtot = np.zeros(self._ngroups)
        _group_tot(val, self._order_le, self._gs_le, self._ge_le, gtot,
                   total_rows)
        win = np.empty(self.n_hero)
        lose = np.empty(self.n_hero)
        tie = np.empty(self.n_hero)
        _accum(le, lt, gtot, self._q_group, self._s_eff, self.n_hero,
               self._n_rows_per_hero, win, lose, tie)
        return win, lose, tie


def disjoint_rank_aggregate(hero_keys, hero_scores, opp_keys, opp_scores, opp_w):
    """One-shot ``(win, lose, tie)``: reach-weighted all-pairs
    comparison, skipping pairs that share a key — exactly.

    ``win[i] = Σ_j (hero_scores[i] > opp_scores[j]) · disjoint(i, j) ·
    opp_w[j]`` (lose/tie analogous), where ``disjoint`` means key sets
    share no element. O((2^K − 1)·N log N) vs the dense O(N·M), exact
    by inclusion–exclusion (see the module docstring).

    Domain envelope: keys are any int64 values, K distinct per row,
    K <= 12 — AND the number of distinct key values V must satisfy
    ``C(V, K)·(K+1) < 2**62`` (the subset encoding must fit in int64;
    validated, with the cap for your K in the error message). A
    52-card deck fits at every K <= 12; K=5 admits ~9,800 distinct
    values.

    Performance shape: the one-shot pays the full topology build
    (~80% of its time, NumPy lexsort work) and CAN LOSE to a dense
    O(N·M) pass at moderate sizes. The asymptotics win in the
    ITERATIVE form: build once via
    ``st = DisjointRankStructure.build(...)``, then ``st.eval(w)`` per
    weight vector is a few segmented passes with no sorting — the
    shape iterative solvers (CFR) need. If you evaluate a single
    weight vector once, benchmark against your dense reference before
    adopting this.
    """
    st = DisjointRankStructure.build(hero_keys, hero_scores, opp_keys, opp_scores)
    return st.eval(opp_w)
