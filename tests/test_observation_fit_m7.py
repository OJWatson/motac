from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu_observed_poisson_approx,
    generate_random_world,
    simulate_hawkes_counts,
)


def test_fit_observed_poisson_approx_recovers_alpha_ballpark() -> None:
    world = generate_random_world(n_locations=5, seed=50, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=6, beta=1.0)

    alpha_true = 0.7
    mu_true = np.linspace(0.05, 0.15, world.n_locations)
    p_detect = 0.6
    false_rate = 0.2

    params = HawkesDiscreteParams(
        mu=mu_true,
        alpha=alpha_true,
        kernel=kernel,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=180, seed=51)

    fit = fit_hawkes_mle_alpha_mu_observed_poisson_approx(
        world=world,
        kernel=kernel,
        y_true_for_history=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=p_detect,
        false_rate=false_rate,
        init_alpha=0.2,
        maxiter=600,
    )

    alpha_hat = float(fit["alpha"])
    assert alpha_hat >= 0.0

    # Approximate likelihood; keep tolerance loose.
    assert abs(alpha_hat - alpha_true) / alpha_true < 0.7

    # Should improve the approximate loglik vs init.
    assert float(fit["loglik"]) >= float(fit["loglik_init"]) - 1e-6
