import numpy as np
import pytest
from numba import njit

from numba_utils import choice, cumulative_sum, insertion_sort
from numba_utils.testing import (
    assert_close,
    assert_converges,
    assert_equivalent,
    assert_no_reweight_bias,
    assert_reproducible,
    assert_within_se,
    deterministic_rng,
    mutation_screams,
    random_arrays,
)


def _py_cumsum(arr):
    return np.cumsum(arr)


class TestAssertClose:
    def test_passes_on_close(self):
        assert_close(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-12]))

    def test_fails_on_far(self):
        with pytest.raises(AssertionError):
            assert_close(1.0, 1.1)


class TestAssertEquivalent:
    def test_validates_matching_kernel(self):
        checked = assert_equivalent(
            _py_cumsum,
            cumulative_sum,
            random_arrays(n_cases=10, size=500, seed=1),
        )
        assert checked == 15  # 10 random + 5 edge cases

    def test_protects_against_mutation(self):
        # insertion_sort mutates its input; np.sort does not. Per-call
        # copies mean the mutating candidate still compares correctly.
        assert_equivalent(
            np.sort,
            insertion_sort,
            random_arrays(n_cases=3, size=50, seed=2),
        )

    def test_reports_failing_case(self):
        def wrong(arr):
            return np.cumsum(arr) + 1.0

        with pytest.raises(AssertionError, match="case 0"):
            assert_equivalent(
                _py_cumsum, wrong, random_arrays(n_cases=1, size=10)
            )

    def test_tuple_results_compared_elementwise(self):
        def ref(arr):
            return arr.min(), arr.max()

        @njit
        def cand(arr):
            lo = arr[0]
            hi = arr[0]
            for x in arr:
                if x < lo:
                    lo = x
                elif x > hi:
                    hi = x
            return lo, hi

        assert_equivalent(ref, cand, random_arrays(n_cases=5, size=100))

    def test_empty_inputs_raise(self):
        with pytest.raises(AssertionError, match="no cases"):
            assert_equivalent(_py_cumsum, cumulative_sum, [])

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            assert_equivalent(42, cumulative_sum, [np.ones(3)])


class TestRandomArrays:
    def test_deterministic_per_seed(self):
        a = list(random_arrays(n_cases=3, size=20, seed=5))
        b = list(random_arrays(n_cases=3, size=20, seed=5))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_edge_cases_present(self):
        cases = list(random_arrays(n_cases=0, size=10))
        assert len(cases) == 5
        assert (cases[0] == cases[0][0]).all()  # constant
        assert (np.diff(cases[1]) >= 0).all()  # ascending
        assert (np.diff(cases[2]) <= 0).all()  # descending
        assert cases[4].shape == (1,)  # single element

    def test_dtype_respected(self):
        for arr in random_arrays(n_cases=2, size=8, dtype=np.int32):
            assert arr.dtype == np.int32

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            list(random_arrays(n_cases=-1, size=10))
        with pytest.raises(ValueError):
            list(random_arrays(n_cases=1, size=0))


class TestDeterministicRng:
    def test_pins_all_three_worlds(self):
        gen_a = deterministic_rng(11)
        legacy_a = np.random.random(3)
        modern_a = gen_a.random(3)
        numba_a = choice(np.arange(100, dtype=np.float64), 10)

        gen_b = deterministic_rng(11)
        np.testing.assert_array_equal(np.random.random(3), legacy_a)
        np.testing.assert_array_equal(gen_b.random(3), modern_a)
        np.testing.assert_array_equal(
            choice(np.arange(100, dtype=np.float64), 10), numba_a
        )


class TestAssertReproducible:
    def test_passes_for_seeded_function(self):
        def draw():
            return choice(np.arange(100, dtype=np.float64), 10)

        result = assert_reproducible(draw, seed=3)
        assert result.shape == (10,)

    def test_fails_for_unseeded_state(self):
        state = {"n": 0}

        def impure():
            state["n"] += 1
            return state["n"]

        with pytest.raises(AssertionError):
            assert_reproducible(impure)

    def test_validation(self):
        with pytest.raises(TypeError):
            assert_reproducible(42)
        with pytest.raises(ValueError):
            assert_reproducible(lambda: 0, runs=1)


class TestAssertConverges:
    def test_passes_for_correct_estimator(self):
        def estimate_mean():
            return float(np.random.random(4000).mean())

        mean, se = assert_converges(estimate_mean, 0.5, n_runs=20)
        assert abs(mean - 0.5) < 0.05
        assert se > 0

    def test_fails_for_biased_estimator(self):
        def biased():
            return float(np.random.random(4000).mean()) + 0.1

        with pytest.raises(AssertionError):
            assert_converges(biased, 0.5, n_runs=20)

    def test_zero_variance_detected(self):
        with pytest.raises(AssertionError, match="identical"):
            assert_converges(lambda: 0.4, 0.5, n_runs=5)
        # zero variance but exactly right is accepted
        mean, se = assert_converges(lambda: 0.5, 0.5, n_runs=5)
        assert (mean, se) == (0.5, 0.0)

    def test_validation(self):
        # n_runs below 5 rejected: the t-statistic false-positive rate
        # explodes (~20% at n_runs=2, ~4% at 5 — documented)
        with pytest.raises(ValueError):
            assert_converges(lambda: 0.0, 0.0, n_runs=4)
        with pytest.raises(ValueError):
            assert_converges(lambda: 0.0, 0.0, sigma=0.0)
        with pytest.raises(ValueError):
            assert_converges(lambda: 0.0, np.inf)
        with pytest.raises(AssertionError, match="non-finite"):
            assert_converges(lambda: np.nan, 0.5, n_runs=5)

    def test_pass_seed_enables_counter_based_kernels(self):
        from numba_utils.random import philox_uniform

        def philox_mean(key):
            acc = 0.0
            for i in range(2000):
                acc += philox_uniform(key, i)
            return acc / 2000

        # without pass_seed, a pure counter-based kernel has zero
        # variance across global seeds and the assert cannot work
        with pytest.raises(AssertionError, match="identical"):
            assert_converges(lambda: philox_mean(1), 0.5, n_runs=5)
        # with pass_seed, each run gets its own key -> real variance
        mean, se = assert_converges(
            philox_mean, 0.5, n_runs=10, pass_seed=True
        )
        assert se > 0
        assert abs(mean - 0.5) < 0.05

    def test_reproducible_pass_seed(self):
        from numba_utils.random import philox_uniform

        def draw(key):
            return philox_uniform(key, 0)

        assert_reproducible(draw, seed=5, pass_seed=True)


class TestAssertWithinSe:
    def test_passes_within_band(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(10.0, 0.5, 100)
        mean, se = assert_within_se(samples, 10.0, k=3.0)
        assert se > 0
        assert abs(mean - 10.0) < 0.2

    def test_fails_far_from_target(self):
        rng = np.random.default_rng(2)
        with pytest.raises(AssertionError):
            assert_within_se(rng.normal(10.0, 0.1, 100), 20.0)

    def test_zero_variance_and_validation(self):
        with pytest.raises(AssertionError, match="identical"):
            assert_within_se([0.4] * 5, 0.5)
        assert assert_within_se([0.5] * 5, 0.5) == (0.5, 0.0)
        with pytest.raises(ValueError):
            assert_within_se([1.0, 2.0], 1.5)  # below the n >= 5 floor
        with pytest.raises(AssertionError, match="non-finite"):
            assert_within_se([1.0, np.nan, 1.0, 1.0, 1.0], 1.0)

    def test_identical_samples_pass_despite_mean_rounding(self):
        # 0.4.0-verdict regression: np.mean over 30 copies of 0.1 is
        # NOT 0.1 (pairwise-summation rounding), the sample std is
        # ~1 ULP instead of 0.0, and an EXACTLY correct value failed
        # at a deterministic ~5.4 SE. Identity must be detected
        # structurally, not via se == 0.0.
        for value in (0.1, 0.2, 0.3, 0.7, 1.1, 3.7):
            assert assert_within_se([value] * 30, value) == (value, 0.0)

    def test_ulp_scale_deviation_is_not_bias(self):
        # A mean within a few ULP of the target must pass even when
        # the measured SE is at rounding-noise scale.
        base = 0.1
        samples = np.full(30, base)
        samples[0] = np.nextafter(base, 1.0)  # break bit-identity
        mean, _se = assert_within_se(samples, base)
        assert mean != base  # the deviation existed; it was ULP-scale


class TestMutationScreams:
    def test_live_check_passes_and_returns_deviation(self):
        def run(broken):
            arr = np.arange(10.0)
            return arr.sum() + (100.0 if broken else 0.0)

        assert mutation_screams(run, threshold=1.0) == 100.0

    def test_dead_check_raises(self):
        # the mutation changes nothing observable -> the check this
        # protects cannot fail -> must raise
        def run(broken):
            return np.arange(10.0).sum()

        with pytest.raises(AssertionError, match="does NOT scream"):
            mutation_screams(run, threshold=1.0)

    def test_non_finite_deviation_counts_as_scream(self):
        def run(broken):
            return np.inf if broken else 1.0

        assert not np.isfinite(mutation_screams(run, threshold=1.0))

    def test_validation(self):
        with pytest.raises(TypeError):
            mutation_screams(42, threshold=1.0)
        with pytest.raises(ValueError):
            mutation_screams(lambda broken: 0.0, threshold=0.0)

    def test_none_result_rejected(self):
        # 0.4.0-verdict regression: an in-place kernel returning None
        # became np.asarray(None) -> NaN -> "non-finite counts as a
        # scream" -> the check certified a mutation wired to NOTHING.
        with pytest.raises(TypeError, match="None"):
            mutation_screams(lambda broken: None, threshold=1.0)

        def in_place_forgot_return(broken):
            buf = np.zeros(4)
            if broken:
                buf[0] = 99.0
            return None  # mutated buf, returned nothing

        with pytest.raises(TypeError, match="RETURN the buffer"):
            mutation_screams(in_place_forgot_return, threshold=1.0)

    def test_identical_nan_outputs_do_not_scream(self):
        # 0.4.0-verdict regression: NaN in the output with the mutation
        # NOT wired (identical outputs) passed as a scream because
        # NaN - NaN = NaN was counted as non-finite deviation.
        def run(broken):
            return np.array([1.0, np.nan, 3.0])  # same either way

        with pytest.raises(AssertionError, match="does NOT scream"):
            mutation_screams(run, threshold=1.0)

    def test_matching_nonfinite_positions_do_not_mask_real_deviation(self):
        # inf/NaN present in BOTH runs at the same positions must not
        # drown a real finite deviation elsewhere.
        def run(broken):
            out = np.array([np.inf, np.nan, 1.0])
            if broken:
                out[2] = 51.0
            return out

        assert mutation_screams(run, threshold=1.0) == 50.0

    def test_nan_appearing_is_a_scream(self):
        def run(broken):
            out = np.array([1.0, 2.0])
            if broken:
                out[1] = np.nan
            return out

        assert not np.isfinite(mutation_screams(run, threshold=1.0))


class TestAssertNoReweightBias:
    def test_correct_estimator_passes(self):
        from numba_utils.stats import weighted_mc_mean

        def correct(values, weights, run_seed):
            return weighted_mc_mean(values, weights, 200, run_seed, 0)

        mean, se = assert_no_reweight_bias(correct)
        assert se > 0

    def test_reach_squared_estimator_screams(self):
        # THE bug: subsample proportional to the weights, then weight
        # again -> effective weight**2
        def broken(values, weights, run_seed):
            rng = np.random.default_rng(run_seed)
            p = weights / weights.sum()
            idx = rng.choice(values.shape[0], 200, replace=False, p=p)
            w = weights[idx]
            return float(np.sum(w * values[idx]) / np.sum(w))

        with pytest.raises(AssertionError):
            assert_no_reweight_bias(broken)

    def test_underpowered_run_is_inconclusive_not_a_pass(self):
        # 0.4.0-verdict regression: the k·SE criterion self-hides — a
        # noisy estimator widens its own tolerance band. The library's
        # own kernel at n_sub=5 has k·SE larger than the gap between
        # the exact and double-weighted means; it used to pass, now it
        # must fail as inconclusive.
        from numba_utils.stats import weighted_mc_mean

        def noisy(values, weights, run_seed):
            return weighted_mc_mean(values, weights, 5, run_seed, 0)

        with pytest.raises(AssertionError, match="INCONCLUSIVE"):
            assert_no_reweight_bias(noisy)
