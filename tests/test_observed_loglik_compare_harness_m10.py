from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    compare_observed_loglik_exact_vs_poisson_approx,
    discrete_exponential_kernel,
    generate_random_world,
    simulate_hawkes_counts,
)


def test_compare_observed_loglik_harness_reports_finite_values() -> None:
    world = generate_random_world(n_locations=3, seed=42, lengthscale=0.6)
    kernel = discrete_exponential_kernel(n_lags=4, beta=1.3)

    params = HawkesDiscreteParams(
        mu=np.linspace(0.05, 0.09, world.n_locations),
        alpha=0.6,
        kernel=kernel,
        p_detect=0.7,
        false_rate=0.2,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=80, seed=5)

    cmp = compare_observed_loglik_exact_vs_poisson_approx(
        world=world,
        kernel=kernel,
        mu=params.mu,
        alpha=params.alpha,
        y_true_for_history=out["y_true"],
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params.p_detect,
        false_rate=params.false_rate,
    )

    assert np.isfinite(cmp.ll_exact)
    assert np.isfinite(cmp.ll_poisson_approx)
    assert np.isfinite(cmp.delta_exact_minus_approx)

    # Exact likelihood conditions on y_true and should typically dominate the
    # Poisson approximation evaluated at the same params.
    assert cmp.ll_exact >= cmp.ll_poisson_approx - 1e-6
