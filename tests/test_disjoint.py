import numpy as np
import pytest

from numba_utils.algorithms import DisjointRankStructure, disjoint_rank_aggregate
from numba_utils.testing import mutation_screams

RNG = np.random.default_rng(23)


def _random_items(n, k, universe, seed):
    rng = np.random.default_rng(seed)
    keys = np.empty((n, k), np.int64)
    for i in range(n):
        keys[i] = rng.choice(universe, k, replace=False)
    # coarse scores so real ties exist
    scores = rng.integers(0, max(4, n // 4), n)
    return keys, scores


def _dense(hero_keys, hero_scores, opp_keys, opp_scores, w, universe):
    """Independent dense reference: explicit overlap mask, O(N*M).
    Shares no code path with the inclusion-exclusion machinery."""
    n, m = len(hero_keys), len(opp_keys)
    hm = np.zeros((n, universe), bool)
    om = np.zeros((m, universe), bool)
    for i in range(n):
        hm[i, hero_keys[i]] = True
    for j in range(m):
        om[j, opp_keys[j]] = True
    disjoint = ~(hm @ om.T)
    diff = hero_scores[:, None] - opp_scores[None, :]
    win = ((diff > 0) & disjoint) @ w
    lose = ((diff < 0) & disjoint) @ w
    tie = ((diff == 0) & disjoint) @ w
    return win, lose, tie


class TestDisjointRankAggregate:
    def test_matches_dense_reference(self):
        # ranges with REAL key overlap so the removal term is exercised
        universe = 20
        hk, hs = _random_items(150, 5, universe, 1)
        ok, os_ = _random_items(180, 5, universe, 2)
        w = np.random.default_rng(3).random(180)
        d_win, d_lose, d_tie = _dense(hk, hs, ok, os_, w, universe)
        win, lose, tie = disjoint_rank_aggregate(hk, hs, ok, os_, w)
        np.testing.assert_allclose(win, d_win, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(lose, d_lose, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(tie, d_tie, rtol=1e-9, atol=1e-12)

    def test_k_generic(self):
        for k, universe in ((1, 8), (2, 10), (3, 12), (7, 30)):
            hk, hs = _random_items(60, k, universe, k)
            ok, os_ = _random_items(70, k, universe, k + 100)
            w = np.random.default_rng(k).random(70)
            d = _dense(hk, hs, ok, os_, w, universe)
            got = disjoint_rank_aggregate(hk, hs, ok, os_, w)
            for a, b in zip(got, d):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_arbitrary_key_values(self):
        # rank compression: keys need not be small or contiguous
        hk = np.array([[10**12, -5], [7, 10**12]], np.int64)
        ok = np.array([[-5, 7], [999, 1000]], np.int64)
        hs = np.array([2, 1])
        os_ = np.array([1, 1])
        w = np.array([1.0, 10.0])
        win, lose, tie = disjoint_rank_aggregate(hk, hs, ok, os_, w)
        # hero 0 {1e12,-5}: opp0 {-5,7} shares -5 -> excluded; opp1
        # disjoint, equal? hero0 score 2 > 1 -> win 10
        assert win[0] == 10.0 and tie[0] == 0.0
        # hero 1 {7,1e12}: opp0 shares 7 -> excluded; opp1 disjoint,
        # scores equal -> tie 10
        assert tie[1] == 10.0 and win[1] == 0.0

    def test_structure_reuse_across_weight_vectors(self):
        universe = 16
        hk, hs = _random_items(80, 4, universe, 5)
        ok, os_ = _random_items(90, 4, universe, 6)
        st = DisjointRankStructure.build(hk, hs, ok, os_)
        rng = np.random.default_rng(7)
        for _ in range(3):
            w = rng.random(90)
            d = _dense(hk, hs, ok, os_, w, universe)
            got = st.eval(w)
            for a, b in zip(got, d):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_win_lose_tie_partition_the_disjoint_mass(self):
        universe = 14
        hk, hs = _random_items(50, 3, universe, 8)
        ok, os_ = _random_items(60, 3, universe, 9)
        w = np.random.default_rng(10).random(60)
        win, lose, tie = disjoint_rank_aggregate(hk, hs, ok, os_, w)
        _, _, _ = win, lose, tie
        # win + lose + tie must equal the total disjoint weight per hero
        hm = np.zeros((50, universe), bool)
        om = np.zeros((60, universe), bool)
        for i in range(50):
            hm[i, hk[i]] = True
        for j in range(60):
            om[j, ok[j]] = True
        expected_mass = (~(hm @ om.T)) @ w
        np.testing.assert_allclose(win + lose + tie, expected_mass, rtol=1e-9)

    def test_mutation_drop_removal_screams(self):
        # certification, ported from the contributing solver: zero the
        # subset query rows' IE signs (keep only the global +1) -> the
        # card-removal term goes dead and the dense reference disagrees
        # by order the blocked-opponent mass. Proves the term is live.
        universe = 18
        hk, hs = _random_items(100, 5, universe, 11)
        ok, os_ = _random_items(100, 5, universe, 12)
        w = np.random.default_rng(13).random(100)
        d_win, _, _ = _dense(hk, hs, ok, os_, w, universe)

        def run(broken):
            st = DisjointRankStructure.build(hk, hs, ok, os_)
            if broken:
                s = st._s_eff.copy()
                s[st.n_hero:] = 0.0  # kill every subset row's IE sign
                st._s_eff = s
            win, _, _ = st.eval(w)
            return np.max(np.abs(win - d_win))

        deviation = mutation_screams(run, threshold=1e-3)
        assert deviation > 1.0  # blocked mass is macroscopic here

    def test_validation(self):
        hk = np.array([[0, 1]], np.int64)
        ok = np.array([[2, 3]], np.int64)
        s = np.array([1])
        w = np.array([1.0])
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(np.array([[0, 0]]), s, ok, s, w)  # dup keys
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(hk, s, np.array([[1, 2, 3]]), np.array([1]), w)
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(hk, np.array([1, 2]), ok, s, w)
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(hk, s, ok, s, np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(hk, s, ok, s, np.array([np.nan]))
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(hk, s, ok, s, np.array([-1.0]))
        with pytest.raises(ValueError):
            disjoint_rank_aggregate(
                np.array([[0.5, 1.5]]), s, ok, s, w
            )  # float keys
        with pytest.raises(ValueError):
            st = DisjointRankStructure.build(
                np.arange(13)[None, :], s, np.arange(13, 26)[None, :], s
            )  # K > 12

    def test_zero_weights_contribute_nothing(self):
        universe = 12
        hk, hs = _random_items(40, 3, universe, 14)
        ok, os_ = _random_items(40, 3, universe, 15)
        w = np.zeros(40)
        win, lose, tie = disjoint_rank_aggregate(hk, hs, ok, os_, w)
        assert not win.any() and not lose.any() and not tie.any()
