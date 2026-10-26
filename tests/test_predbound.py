import math

import numpy as np

from predbound import bound


def test_fano_inversion_endpoints_and_monotone():
    assert abs(bound.fano_pi(0.0, 10) - 1.0) < 1e-6
    assert abs(bound.fano_pi(math.log2(10), 10) - 0.1) < 1e-9
    assert bound.fano_pi(1.0, 10) > bound.fano_pi(2.0, 10)


def test_deterministic_cycle_has_zero_entropy_and_perfect_prediction():
    P = np.roll(np.eye(5), 1, axis=1)
    pi, H, best = bound.markov_entropy_and_best(P)
    assert H < 1e-9 and abs(best - 1) < 1e-9


def test_lz_entropy_is_low_for_a_repeating_sequence_and_high_for_random_noise():
    rng = np.random.default_rng(0)
    rep = [0, 1, 2, 3] * 250
    rnd = [int(x) for x in rng.integers(0, 4, 1000)]
    assert bound.lz_entropy(rep) < 0.5 and 1.4 < bound.lz_entropy(rnd) < 2.6


def test_bound_is_never_below_the_best_first_order_accuracy_for_exact_entropy():
    rng = np.random.default_rng(3)
    for _ in range(5):
        P = bound.routine_matrix(12, rng, stay=0.4)
        _, H, best = bound.markov_entropy_and_best(P)
        assert bound.fano_pi(H, 12) >= best - 1e-6
