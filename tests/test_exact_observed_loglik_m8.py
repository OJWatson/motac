from __future__ import annotations

import numpy as np

from motac.sim import (
    discrete_exponential_kernel,
    generate_random_world,
    hawkes_loglik_observed_exact,
)


def test_exact_observed_loglik_matches_hand_computable_toy() -> None:
    # Single location, single time point.
    # y_true = 2, y_obs = 1, p=0.5, false_rate=1.
    # p(y_obs=1|y_true=2) = sum_{k=0..1} Binom(k|2,0.5)*Pois(1-k|1)
    # = Binom(0)*Pois(1) + Binom(1)*Pois(0)
    # = 0.25*e^-1*1 + 0.5*e^-1
    # = 0.75/e
    world = generate_random_world(n_locations=1, seed=0, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=2, beta=1.0)

    mu = np.array([0.1])
    alpha = 0.0

    y_true = np.array([[2]], dtype=int)
    y_obs = np.array([[1]], dtype=int)

    ll = hawkes_loglik_observed_exact(
        world=world,
        kernel=kernel,
        mu=mu,
        alpha=alpha,
        y_true_for_history=y_true,
        y_true=y_true,
        y_obs=y_obs,
        p_detect=0.5,
        false_rate=1.0,
    )

    expected = float(np.log(0.75) - 1.0)
    assert np.isfinite(ll)
    assert abs(ll - expected) < 1e-10
