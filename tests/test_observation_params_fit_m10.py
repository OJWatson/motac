from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_observation_params_exact,
    generate_random_world,
    simulate_hawkes_counts,
)


def test_fit_observation_params_exact_recovers_ballpark() -> None:
    world = generate_random_world(n_locations=4, seed=0, lengthscale=0.5)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.1),
        alpha=0.4,
        kernel=discrete_exponential_kernel(n_lags=4, beta=1.0),
        p_detect=0.65,
        false_rate=0.2,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=250, seed=1)

    fit = fit_observation_params_exact(
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        init_p_detect=0.8,
        init_false_rate=0.05,
        maxiter=400,
    )

    p_hat = float(fit["p_detect"])
    fr_hat = float(fit["false_rate"])

    assert 0.0 < p_hat < 1.0
    assert fr_hat >= 0.0

    # Recovery is noisy; check ballpark.
    assert abs(p_hat - params.p_detect) < 0.2
    assert abs(fr_hat - params.false_rate) < 0.2

    # Likelihood should improve vs init.
    assert float(fit["loglik"]) >= float(fit["loglik_init"]) - 1e-6
