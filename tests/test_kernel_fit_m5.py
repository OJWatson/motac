from __future__ import annotations

import numpy as np
import pytest

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu_beta,
    generate_random_world,
    predict_hawkes_intensity_multi_step,
    simulate_hawkes_counts,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mle_recovers_beta_alpha_reasonable_across_seeds(seed: int) -> None:
    world = generate_random_world(n_locations=6, seed=100 + seed, lengthscale=0.5)

    n_lags = 6
    beta_true = 0.9
    kernel = discrete_exponential_kernel(n_lags=n_lags, beta=beta_true)

    alpha_true = 0.6
    mu_true = np.linspace(0.05, 0.15, world.n_locations)
    params = HawkesDiscreteParams(mu=mu_true, alpha=alpha_true, kernel=kernel)

    out = simulate_hawkes_counts(world=world, params=params, n_steps=160, seed=200 + seed)
    y = out["y_true"]

    fit = fit_hawkes_mle_alpha_mu_beta(
        world=world,
        n_lags=n_lags,
        y=y,
        init_alpha=0.2,
        init_beta=0.5,
        maxiter=600,
    )

    alpha_hat = float(fit["alpha"])
    beta_hat = float(fit["beta"])

    assert alpha_hat >= 0.0
    assert beta_hat > 0.0

    # Loose recovery checks (stochastic + partial identifiability with alpha).
    assert abs(alpha_hat - alpha_true) / alpha_true < 0.6

    # Beta itself can be weakly identified; check the implied kernel shape is close.
    kernel_hat = np.asarray(fit["kernel"], dtype=float)
    kernel_true = kernel
    corr = float(np.corrcoef(kernel_true, kernel_hat)[0, 1])
    assert corr > 0.9


def test_fit_improves_loglik_vs_init() -> None:
    world = generate_random_world(n_locations=5, seed=3, lengthscale=0.4)

    n_lags = 5
    beta_true = 1.2
    kernel = discrete_exponential_kernel(n_lags=n_lags, beta=beta_true)

    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.08),
        alpha=0.7,
        kernel=kernel,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=120, seed=4)
    y = out["y_true"]

    fit = fit_hawkes_mle_alpha_mu_beta(
        world=world,
        n_lags=n_lags,
        y=y,
        init_alpha=0.05,
        init_beta=0.2,
        maxiter=500,
    )

    assert float(fit["loglik"]) >= float(fit["loglik_init"]) - 1e-6


def test_forecast_stability_sanity() -> None:
    world = generate_random_world(n_locations=4, seed=11, lengthscale=0.6)

    n_lags = 4
    beta_true = 0.8
    kernel = discrete_exponential_kernel(n_lags=n_lags, beta=beta_true)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.05),
        alpha=0.5,
        kernel=kernel,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=60, seed=12)
    y_hist = out["y_true"][:, :40]

    # Fit beta/alpha/mu and then forecast deterministically.
    fit = fit_hawkes_mle_alpha_mu_beta(world=world, n_lags=n_lags, y=y_hist, maxiter=400)
    params_hat = HawkesDiscreteParams(
        mu=np.asarray(fit["mu"], dtype=float),
        alpha=float(fit["alpha"]),
        kernel=np.asarray(fit["kernel"], dtype=float),
    )

    lam = predict_hawkes_intensity_multi_step(
        world=world, params=params_hat, y_history=y_hist, horizon=25
    )

    assert lam.shape == (world.n_locations, 25)
    assert np.all(np.isfinite(lam))
    assert np.all(lam >= 0.0)
    # Avoid explosive nonsense in this small test.
    assert float(lam.max()) < 1e4
