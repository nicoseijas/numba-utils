import numpy as np
import pytest
from numba import njit

from numba_utils.random import (
    alias_draw,
    alias_sample,
    alias_setup,
    choice,
    partial_shuffle,
    permutation,
    philox4x64,
    philox_partial_shuffle,
    philox_randint,
    philox_sample_without_replacement,
    philox_uniform,
    philox_uniforms,
    reservoir_sampling,
    sample_without_replacement,
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


class TestPhilox:
    def test_block_matches_numpy_philox_exactly(self):
        # NumPy increments the counter BEFORE generating, so its first
        # raw block for counter=[c,0,0,0] is our block at counter c+1.
        # Values >= 2**63 go through np.uint64: Numba's dispatcher
        # types plain Python ints as int64.
        for key, ctr in [(0, 0), (42, 123), (2**63, 2**40), (7, 2**62)]:
            ref = np.random.Philox(counter=[ctr, 0, 0, 0], key=[key, 0])
            expected = ref.random_raw(8).tolist()
            k = np.uint64(key)
            ours = list(philox4x64(k, 0, np.uint64(ctr + 1), 0, 0, 0)) + list(
                philox4x64(k, 0, np.uint64(ctr + 2), 0, 0, 0)
            )
            assert [int(x) for x in ours] == expected

    def test_uniform_range_and_determinism(self):
        values = [philox_uniform(9, c) for c in range(1000)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert values == [philox_uniform(9, c) for c in range(1000)]
        assert 0.4 < np.mean(values) < 0.6

    def test_streams_are_independent_of_call_order(self):
        forward = [philox_uniform(3, c) for c in range(50)]
        backward = [philox_uniform(3, c) for c in reversed(range(50))]
        assert forward == backward[::-1]

    def test_different_keys_differ(self):
        a = [philox_uniform(1, c) for c in range(100)]
        b = [philox_uniform(2, c) for c in range(100)]
        assert a != b

    def test_uniforms_split_across_counters(self):
        # consuming ceil(size/4) blocks: [ctr, 4 values] + [ctr+1, 4]
        # must equal one call of 8 starting at ctr
        whole = philox_uniforms(5, 100, 8)
        first = philox_uniforms(5, 100, 4)
        second = philox_uniforms(5, 101, 4)
        np.testing.assert_array_equal(whole, np.concatenate([first, second]))

    def test_uniforms_out_and_errors(self):
        out = np.empty(6)
        result = philox_uniforms(1, 0, 6, out)
        assert result is out
        with pytest.raises(ValueError):
            philox_uniforms(1, 0, -1)
        with pytest.raises(ValueError):
            philox_uniforms(1, 0, 3, np.empty(4))

    def test_randint_range_and_errors(self):
        draws = [philox_randint(11, c, 52) for c in range(2000)]
        assert all(0 <= d < 52 for d in draws)
        assert len(set(draws)) == 52
        assert philox_randint(11, 7, 1) == 0
        with pytest.raises(ValueError):
            philox_randint(11, 0, 0)

    def test_callable_from_jitted_code(self):
        @njit
        def mc_mean(key, n):
            acc = 0.0
            for i in range(n):
                acc += philox_uniform(key, i)
            return acc / n

        assert 0.45 < mc_mean(21, 10_000) < 0.55
        assert mc_mean(21, 10_000) == mc_mean(21, 10_000)


class TestPartialShuffle:
    def test_prefix_is_sample_without_replacement(self):
        seed(3)
        arr = np.arange(52, dtype=np.int64)
        partial_shuffle(arr, 5)
        assert len(set(arr[:5].tolist())) == 5
        assert sorted(arr.tolist()) == list(range(52))

    def test_k_zero_is_noop(self):
        arr = np.arange(5, dtype=np.int64)
        partial_shuffle(arr, 0)
        np.testing.assert_array_equal(arr, np.arange(5))

    def test_uniformity_of_first_slot(self):
        seed(11)
        counts = np.zeros(4, np.int64)
        for _ in range(4000):
            arr = np.arange(4, dtype=np.int64)
            partial_shuffle(arr, 1)
            counts[arr[0]] += 1
        assert counts.min() > 800  # expected 1000 each

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            partial_shuffle(np.arange(3, dtype=np.int64), 4)


class TestSampleWithoutReplacement:
    def test_no_duplicates_and_input_untouched(self):
        seed(5)
        arr = np.arange(52, dtype=np.int64)
        out = sample_without_replacement(arr, 7)
        assert out.shape == (7,)
        assert len(set(out.tolist())) == 7
        np.testing.assert_array_equal(arr, np.arange(52))

    def test_full_draw_is_permutation(self):
        seed(9)
        out = sample_without_replacement(np.arange(10, dtype=np.int64), 10)
        assert sorted(out.tolist()) == list(range(10))

    def test_k_out_of_range_raises(self):
        arr = np.arange(3, dtype=np.int64)
        with pytest.raises(ValueError):
            sample_without_replacement(arr, 0)
        with pytest.raises(ValueError):
            sample_without_replacement(arr, 4)


class TestPhiloxSampling:
    def test_partial_shuffle_reproducible_and_valid(self):
        a = np.arange(52, dtype=np.int64)
        b = np.arange(52, dtype=np.int64)
        philox_partial_shuffle(a, 5, 7, 100)
        philox_partial_shuffle(b, 5, 7, 100)
        np.testing.assert_array_equal(a, b)
        assert len(set(a[:5].tolist())) == 5
        assert sorted(a.tolist()) == list(range(52))

    def test_disjoint_counters_differ(self):
        a = philox_sample_without_replacement(
            np.arange(52, dtype=np.int64), 5, 7, 0
        )
        b = philox_sample_without_replacement(
            np.arange(52, dtype=np.int64), 5, 7, 5
        )
        assert a.tolist() != b.tolist()

    def test_sample_no_duplicates_input_untouched(self):
        deck = np.arange(52, dtype=np.int64)
        out = philox_sample_without_replacement(deck, 7, 3, 0)
        assert len(set(out.tolist())) == 7
        np.testing.assert_array_equal(deck, np.arange(52))

    def test_uniformity_of_first_slot(self):
        counts = np.zeros(4, np.int64)
        for it in range(4000):
            arr = np.arange(4, dtype=np.int64)
            philox_partial_shuffle(arr, 1, 13, it)
            counts[arr[0]] += 1
        assert counts.min() > 800

    def test_errors(self):
        with pytest.raises(ValueError):
            philox_partial_shuffle(np.arange(3, dtype=np.int64), 4, 0, 0)
        with pytest.raises(ValueError):
            philox_sample_without_replacement(
                np.arange(3, dtype=np.int64), 0, 0, 0
            )

    def test_callable_from_jitted_code(self):
        @njit
        def deal(deck, k, key, it):
            scratch = deck.copy()
            philox_partial_shuffle(scratch, k, key, it * k)
            return scratch[:k]

        deck = np.arange(52, dtype=np.int64)
        first = deal(deck, 5, 7, 0)
        again = deal(deck, 5, 7, 0)
        np.testing.assert_array_equal(first, again)


class TestPhiloxWordIndependence:
    def test_uniform_and_randint_not_correlated(self):
        # same (key, counter): randint uses word x1, uniform uses x0 —
        # they must NOT be the same underlying number
        n = 1_000_000
        mismatches = sum(
            philox_randint(5, c, n) != int(philox_uniform(5, c) * n)
            for c in range(200)
        )
        assert mismatches > 150


class TestAliasDrawValidation:
    def test_mismatched_tables_raise(self):
        prob, alias = alias_setup(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            alias_draw(prob, alias[:2])


class TestPhiloxUniformsOutShape:
    def test_2d_out_fails_loudly_at_compile(self):
        # a non-1-D out can't unify with the internal allocation; the
        # failure is a loud TypingError, never silent corruption
        from numba.core.errors import TypingError

        with pytest.raises(TypingError):
            philox_uniforms(1, 0, 4, np.empty((4, 1)))


class Test2DInputsRejected:
    def test_swap_based_functions_reject_2d(self):
        # On a 2-D array the element swap works on row VIEWS, which
        # alias: rows get silently duplicated — a deck dealing the
        # same card twice, plausible to the eye. Must raise instead.
        deck2d = np.arange(20, dtype=np.int64).reshape(10, 2)
        with pytest.raises(ValueError):
            shuffle(deck2d.copy())
        with pytest.raises(ValueError):
            partial_shuffle(deck2d.copy(), 4)
        with pytest.raises(ValueError):
            philox_partial_shuffle(deck2d.copy(), 4, 7, 0)
        with pytest.raises(ValueError):
            sample_without_replacement(deck2d, 4)
        with pytest.raises(ValueError):
            philox_sample_without_replacement(deck2d, 4, 7, 0)


class TestPhiloxUniformityOfPrefixes:
    def test_ordered_pairs_equifrequent(self):
        # stronger than the first-slot check: all 20 ordered pairs of
        # (n=5, k=2) must be equifrequent; would catch a biased or
        # mis-indexed step beyond the first swap
        counts = {}
        for it in range(20_000):
            arr = np.arange(5, dtype=np.int64)
            philox_partial_shuffle(arr, 2, 31, it * 2)
            pair = (int(arr[0]), int(arr[1]))
            counts[pair] = counts.get(pair, 0) + 1
        assert len(counts) == 20
        # expected 1000 per pair, sd ~31; +/-150 is ~5 sigma per cell
        assert min(counts.values()) > 850
        assert max(counts.values()) < 1150


class TestPhiloxUniformsWordOverlap:
    def test_overlap_with_randint_is_pinned(self):
        # philox_uniforms consumes ALL FOUR block words: its element
        # [1] is the same word (x1) philox_randint uses at that
        # counter. The counter space is global across philox_* — this
        # test keeps the overlap visible (module docstring tells users
        # to partition counters per purpose).
        for c in range(20):
            u1 = philox_uniforms(9, c, 4)[1]
            assert philox_randint(9, c, 1 << 53) == int(u1 * (1 << 53))
