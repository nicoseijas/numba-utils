import numpy as np
import pytest
from numba import njit

from numba_utils.stats import logsumexp, softmax, weighted_quantile

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
