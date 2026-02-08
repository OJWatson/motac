from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    hawkes_loglik_poisson_observed,
    simulate_hawkes_counts,
)


def test_observed_loglik_prefers_true_params_under_detection_and_clutter() -> None:
    world = generate_random_world(n_locations=5, seed=41, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=6, beta=0.9)

    params_true = HawkesDiscreteParams(
        mu=np.linspace(0.05, 0.15, world.n_locations),
        alpha=0.7,
        kernel=kernel,
        p_detect=0.6,
        false_rate=0.2,
    )

    out = simulate_hawkes_counts(world=world, params=params_true, n_steps=120, seed=42)

    ll_true = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=params_true.mu,
        alpha=params_true.alpha,
        y_true_for_history=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    ll_bad = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=np.full((world.n_locations,), 0.02),
        alpha=0.05,
        y_true_for_history=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    assert np.isfinite(ll_true)
    assert np.isfinite(ll_bad)
    assert ll_true > ll_bad
