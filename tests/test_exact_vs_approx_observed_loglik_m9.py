from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    hawkes_loglik_observed_exact,
    hawkes_loglik_poisson_observed,
    simulate_hawkes_counts,
)


def test_exact_vs_poisson_approx_observed_loglik_on_simulator_data() -> None:
    world = generate_random_world(n_locations=4, seed=123, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=5, beta=1.0)

    params_true = HawkesDiscreteParams(
        mu=np.linspace(0.05, 0.12, world.n_locations),
        alpha=0.7,
        kernel=kernel,
        p_detect=0.6,
        false_rate=0.3,
    )

    out = simulate_hawkes_counts(world=world, params=params_true, n_steps=120, seed=7)

    ll_exact = hawkes_loglik_observed_exact(
        world=world,
        kernel=kernel,
        mu=params_true.mu,
        alpha=params_true.alpha,
        y_true_for_history=out["y_true"],
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    ll_approx = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=params_true.mu,
        alpha=params_true.alpha,
        y_true_for_history=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    assert np.isfinite(ll_exact)
    assert np.isfinite(ll_approx)

    # Exact likelihood conditions on y_true, so it should typically be at least as
    # large as the cheap Poisson approximation evaluated at the same params.
    assert ll_exact >= ll_approx - 1e-6

    # Exact observed likelihood depends on (y_true, y_obs, p_detect, false_rate)
    # and is invariant to Hawkes parameters (mu/alpha/kernel) once y_true is fixed.
    ll_exact_bad = hawkes_loglik_observed_exact(
        world=world,
        kernel=kernel,
        mu=np.full((world.n_locations,), 0.02),
        alpha=0.05,
        y_true_for_history=out["y_true"],
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    assert abs(ll_exact - ll_exact_bad) < 1e-12

    # The Poisson-approx likelihood *does* depend on Hawkes parameters via lambda(t),
    # so it should prefer the true params to a deliberately bad baseline.
    ll_approx_bad = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=np.full((world.n_locations,), 0.02),
        alpha=0.05,
        y_true_for_history=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params_true.p_detect,
        false_rate=params_true.false_rate,
    )

    assert ll_approx > ll_approx_bad
