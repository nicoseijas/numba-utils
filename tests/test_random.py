import numpy as np
import pytest
from numba import njit

from numba_utils.random import (
    alias_draw,
    alias_sample,
    alias_setup,
    choice,
    permutation,
    reservoir_sampling,
    seed,
    shuffle,
    weighted_sampling,
)


class TestSeedDeterminism:
    def test_same_seed_same_stream(self):
        arr = np.arange(100, dtype=np.float64)
        seed(7)
        first = choice(arr, 50)
        seed(7)
        second = choice(arr, 50)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_different_stream(self):
        arr = np.arange(100, dtype=np.float64)
        seed(7)
        first = choice(arr, 50)
        seed(8)
        second = choice(arr, 50)
        assert not np.array_equal(first, second)


class TestShuffle:
    def test_is_permutation_in_place(self):
        seed(1)
        arr = np.arange(1000, dtype=np.float64)
        result = shuffle(arr)
        assert result is arr
        np.testing.assert_array_equal(np.sort(arr), np.arange(1000))

    def test_actually_shuffles(self):
        seed(1)
        arr = np.arange(1000, dtype=np.float64)
        shuffle(arr)
        assert not np.array_equal(arr, np.arange(1000))

    def test_single_and_empty(self):
        np.testing.assert_array_equal(shuffle(np.array([5.0])), [5.0])
        assert shuffle(np.empty(0)).shape == (0,)


class TestPermutation:
    def test_contains_all_indices(self):
        seed(2)
        result = permutation(500)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(np.sort(result), np.arange(500))

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            permutation(-1)


class TestChoice:
    def test_values_come_from_input(self):
        seed(3)
        arr = np.array([10.0, 20.0, 30.0])
        result = choice(arr, 100)
        assert result.shape == (100,)
        assert set(result.tolist()) <= {10.0, 20.0, 30.0}

    def test_dtype_preserved(self):
        seed(3)
        assert choice(np.array([1, 2], dtype=np.int32), 5).dtype == np.int32

    def test_errors(self):
        with pytest.raises(ValueError):
            choice(np.empty(0), 1)
        with pytest.raises(ValueError):
            choice(np.ones(3), -1)


class TestReservoirSampling:
    def test_distinct_positions(self):
        seed(4)
        arr = np.arange(1000, dtype=np.int64)
        sample = reservoir_sampling(arr, 100)
        assert sample.shape == (100,)
        assert len(set(sample.tolist())) == 100

    def test_k_equals_n_is_whole_array(self):
        seed(4)
        arr = np.arange(50, dtype=np.int64)
        sample = reservoir_sampling(arr, 50)
        np.testing.assert_array_equal(np.sort(sample), arr)

    def test_roughly_uniform(self):
        seed(4)
        hits = np.zeros(10)
        arr = np.arange(10, dtype=np.int64)
        for _ in range(2000):
            for v in reservoir_sampling(arr, 3):
                hits[v] += 1
        expected = 2000 * 3 / 10
        assert np.all(hits > expected * 0.8)
        assert np.all(hits < expected * 1.2)

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            reservoir_sampling(np.ones(3), 4)
        with pytest.raises(ValueError):
            reservoir_sampling(np.ones(3), 0)


class TestWeightedSampling:
    def test_respects_proportions(self):
        seed(5)
        weights = np.array([1.0, 0.0, 3.0])
        draws = weighted_sampling(weights, 20_000)
        counts = np.bincount(draws, minlength=3)
        assert counts[1] == 0
        ratio = counts[2] / counts[0]
        assert 2.6 < ratio < 3.4

    def test_errors(self):
        with pytest.raises(ValueError):
            weighted_sampling(np.empty(0), 1)
        with pytest.raises(ValueError):
            weighted_sampling(np.array([1.0, -1.0]), 1)
        with pytest.raises(ValueError):
            weighted_sampling(np.zeros(3), 1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_weight_raises(self, bad):
        with pytest.raises(ValueError):
            weighted_sampling(np.array([1.0, bad]), 5)

    def test_overflowing_sum_raises(self):
        huge = np.finfo(np.float64).max
        with pytest.raises(ValueError):
            weighted_sampling(np.array([huge, huge]), 5)


class TestAlias:
    def test_matches_weighted_proportions(self):
        seed(6)
        weights = np.array([1.0, 0.0, 3.0, 4.0])
        prob, alias = alias_setup(weights)
        draws = alias_sample(prob, alias, 40_000)
        counts = np.bincount(draws, minlength=4)
        assert counts[1] == 0
        total = counts.sum()
        np.testing.assert_allclose(
            counts / total, np.array([1.0, 0.0, 3.0, 4.0]) / 8.0, atol=0.02
        )

    def test_draw_returns_valid_index(self):
        seed(6)
        prob, alias = alias_setup(np.array([2.0, 1.0]))
        for _ in range(50):
            assert alias_draw(prob, alias) in (0, 1)

    def test_uniform_weights_give_prob_one(self):
        prob, _ = alias_setup(np.ones(8))
        np.testing.assert_allclose(prob, np.ones(8))

    def test_errors(self):
        with pytest.raises(ValueError):
            alias_setup(np.empty(0))
        with pytest.raises(ValueError):
            alias_setup(np.array([-1.0, 2.0]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_weight_raises(self, bad):
        with pytest.raises(ValueError):
            alias_setup(np.array([1.0, bad]))

    def test_overflowing_sum_raises(self):
        huge = np.finfo(np.float64).max
        with pytest.raises(ValueError):
            alias_setup(np.array([huge, huge]))

    def test_callable_from_jitted_code(self):
        @njit
        def draw_many(weights, size):
            prob, alias = alias_setup(weights)
            return alias_sample(prob, alias, size)

        seed(9)
        draws = draw_many(np.array([1.0, 1.0]), 10)
        assert draws.shape == (10,)
