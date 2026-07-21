import numpy as np
import pytest
from numba import njit

from numba_utils.stats import (
    logsumexp,
    softmax,
    weighted_mc_mean,
    weighted_quantile,
)

RNG = np.random.default_rng(17)


class TestLogsumexp:
    def test_matches_naive_in_safe_range(self):
        arr = RNG.normal(0.0, 3.0, 10_000)
        expected = np.log(np.sum(np.exp(arr)))
        assert logsumexp(arr) == pytest.approx(expected, rel=1e-12)

    def test_stable_where_naive_overflows(self):
        arr = np.array([1000.0, 1000.0])
        with np.errstate(over="ignore"):
            naive = np.log(np.sum(np.exp(arr)))  # inf
        assert not np.isfinite(naive)
        assert logsumexp(arr) == pytest.approx(1000.0 + np.log(2.0))

    def test_stable_for_large_negative(self):
        arr = np.array([-1000.0, -1000.0])
        assert logsumexp(arr) == pytest.approx(-1000.0 + np.log(2.0))

    def test_edge_values(self):
        assert logsumexp(np.empty(0)) == -np.inf
        assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
        assert logsumexp(np.array([np.inf, 0.0])) == np.inf
        assert np.isnan(logsumexp(np.array([np.nan, 1.0])))

    def test_single_element(self):
        assert logsumexp(np.array([3.5])) == pytest.approx(3.5)

    def test_int_input(self):
        assert logsumexp(np.array([0, 0], np.int64)) == pytest.approx(
            np.log(2.0)
        )


class TestSoftmax:
    def test_matches_reference(self):
        arr = RNG.normal(0.0, 2.0, 5_000)
        e = np.exp(arr - arr.max())
        np.testing.assert_allclose(softmax(arr), e / e.sum(), rtol=1e-12)

    def test_sums_to_one_where_naive_overflows(self):
        arr = np.array([1000.0, 999.0, 998.0])
        result = softmax(arr)
        assert np.all(np.isfinite(result))
        assert result.sum() == pytest.approx(1.0)
        assert result[0] > result[1] > result[2]

    def test_out_buffer_reused(self):
        arr = RNG.normal(0.0, 1.0, 100)
        out = np.empty(100)
        result = softmax(arr, out)
        assert result is out

    def test_errors(self):
        with pytest.raises(ValueError):
            softmax(np.empty(0))
        with pytest.raises(ValueError):
            softmax(np.ones(3), np.empty(4))


class TestWeightedQuantile:
    def test_matches_numpy_inverted_cdf(self):
        values = RNG.normal(0.0, 10.0, 3_000)
        weights = RNG.uniform(0.0, 5.0, 3_000)
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            expected = np.quantile(
                values, q, weights=weights, method="inverted_cdf"
            )
            assert weighted_quantile(values, weights, q) == expected

    def test_uniform_weights_match_unweighted(self):
        values = RNG.normal(0.0, 1.0, 1_000)
        weights = np.ones(1_000)
        for q in (0.25, 0.5, 0.9):
            assert weighted_quantile(values, weights, q) == np.quantile(
                values, q, method="inverted_cdf"
            )

    def test_hand_case(self):
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 2.0])
        assert weighted_quantile(values, weights, 0.5) == 20.0
        assert weighted_quantile(values, weights, 0.75) == 30.0
        assert weighted_quantile(values, weights, 1.0) == 30.0

    def test_zero_weight_not_selected_for_positive_q(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 1.0, 1.0])
        assert weighted_quantile(values, weights, 0.5) == 2.0

    def test_errors(self):
        v = np.array([1.0, 2.0])
        w = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            weighted_quantile(np.empty(0), np.empty(0), 0.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, np.array([1.0]), 0.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, w, 1.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, w, -0.1)
        with pytest.raises(ValueError):
            weighted_quantile(np.array([1.0, np.nan]), w, 0.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, np.array([1.0, np.nan]), 0.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, np.array([1.0, -1.0]), 0.5)
        with pytest.raises(ValueError):
            weighted_quantile(v, np.array([0.0, 0.0]), 0.5)


class TestInsideNjit:
    def test_callable_from_jitted_code(self):
        @njit
        def normalize_scores(scores, weights):
            probs = softmax(scores)
            return logsumexp(scores), weighted_quantile(
                scores, weights, 0.5
            ), probs

        scores = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        lse, med, probs = normalize_scores(scores, weights)
        assert lse == pytest.approx(np.log(np.sum(np.exp(scores))))
        assert med == 2.0
        assert probs.sum() == pytest.approx(1.0)


class TestWeightedQuantileZeroWeightEdge:
    def test_q_zero_skips_zero_weight_minimum(self):
        # NumPy's weighted inverted_cdf at q=0 returns the smallest
        # POSITIVELY-weighted value, not the raw minimum — verified,
        # and we match it (a previous version got this exactly wrong).
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([0.0, 1.0, 1.0])
        expected = np.quantile(v, 0.0, weights=w, method="inverted_cdf")
        assert expected == 2.0
        assert weighted_quantile(v, w, 0.0) == expected


class TestWeightedMcMean:
    def test_exact_when_support_fits(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 0.0, 3.0, 2.0])
        exact = np.sum(weights * values) / np.sum(weights)
        assert weighted_mc_mean(values, weights, 10, 7, 0) == exact

    def test_estimates_the_weighted_mean(self):
        rng = np.random.default_rng(3)
        values = rng.normal(5.0, 2.0, 5000)
        weights = rng.random(5000)
        exact = np.sum(weights * values) / np.sum(weights)
        estimates = [
            weighted_mc_mean(values, weights, 500, key, 0)
            for key in range(30)
        ]
        from numba_utils.testing import assert_within_se

        assert_within_se(estimates, exact, k=4.0)

    def test_reproducible_and_stream_dependent(self):
        rng = np.random.default_rng(4)
        values = rng.normal(0.0, 1.0, 1000)
        weights = rng.random(1000)
        a = weighted_mc_mean(values, weights, 100, 9, 0)
        b = weighted_mc_mean(values, weights, 100, 9, 0)
        c = weighted_mc_mean(values, weights, 100, 10, 0)
        assert a == b
        assert a != c

    def test_validation(self):
        v = np.array([1.0, 2.0])
        w = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            weighted_mc_mean(np.empty(0), np.empty(0), 1, 0, 0)
        with pytest.raises(ValueError):
            weighted_mc_mean(v, np.array([1.0]), 1, 0, 0)
        with pytest.raises(ValueError):
            weighted_mc_mean(v, w, 0, 0, 0)
        with pytest.raises(ValueError):
            weighted_mc_mean(np.array([1.0, np.nan]), w, 1, 0, 0)
        with pytest.raises(ValueError):
            weighted_mc_mean(v, np.array([1.0, -1.0]), 1, 0, 0)
        with pytest.raises(ValueError):
            weighted_mc_mean(v, np.array([0.0, 0.0]), 1, 0, 0)

    def test_callable_from_jitted_code(self):
        @njit
        def estimate(values, weights, key):
            return weighted_mc_mean(values, weights, 50, key, 0)

        rng = np.random.default_rng(5)
        values = rng.normal(0.0, 1.0, 500)
        weights = rng.random(500)
        assert estimate(values, weights, 1) == weighted_mc_mean(
            values, weights, 50, 1, 0
        )
