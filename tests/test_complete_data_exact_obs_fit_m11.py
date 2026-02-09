from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu,
    fit_hawkes_mle_alpha_mu_complete_data_with_exact_obs,
    generate_random_world,
    hawkes_loglik_observed_exact,
    hawkes_loglik_poisson,
    simulate_hawkes_counts,
)


def test_complete_data_wrapper_matches_latent_fit_and_accounts_joint_loglik() -> None:
    world = generate_random_world(n_locations=4, seed=10, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=5, beta=0.9)

    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.1),
        alpha=0.6,
        kernel=kernel,
        p_detect=0.7,
        false_rate=0.2,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=120, seed=11)

    fit_latent = fit_hawkes_mle_alpha_mu(world=world, kernel=kernel, y=out["y_true"], maxiter=200)
    fit_wrap = fit_hawkes_mle_alpha_mu_complete_data_with_exact_obs(
        world=world,
        kernel=kernel,
        y_true_for_history=out["y_true"],
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params.p_detect,
        false_rate=params.false_rate,
        maxiter=200,
    )

    assert np.allclose(np.asarray(fit_wrap["mu"]), np.asarray(fit_latent["mu"]))
    assert float(fit_wrap["alpha"]) == float(fit_latent["alpha"])

    ll_latent = hawkes_loglik_poisson(
        world=world,
        kernel=kernel,
        mu=np.asarray(fit_latent["mu"], dtype=float),
        alpha=float(fit_latent["alpha"]),
        y=out["y_true"],
    )
    ll_obs = hawkes_loglik_observed_exact(
        world=world,
        kernel=kernel,
        mu=np.asarray(fit_latent["mu"], dtype=float),
        alpha=float(fit_latent["alpha"]),
        y_true_for_history=out["y_true"],
        y_true=out["y_true"],
        y_obs=out["y_obs"],
        p_detect=params.p_detect,
        false_rate=params.false_rate,
    )

    assert abs(float(fit_wrap["loglik_latent"]) - float(ll_latent)) < 1e-8
    assert abs(float(fit_wrap["loglik_obs_exact"]) - float(ll_obs)) < 1e-8
    assert abs(float(fit_wrap["loglik_joint"]) - float(ll_latent + ll_obs)) < 1e-8
