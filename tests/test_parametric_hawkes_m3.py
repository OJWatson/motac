from __future__ import annotations

import numpy as np

from motac.sim import (
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu,
    generate_random_world,
    hawkes_intensity,
    hawkes_loglik_poisson,
    simulate_hawkes_counts,
)
from motac.sim.hawkes import HawkesDiscreteParams


def test_loglik_finite_and_prefers_true_params() -> None:
    world = generate_random_world(n_locations=4, seed=123, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=5, beta=0.9)

    params_true = HawkesDiscreteParams(
        mu=np.array([0.08, 0.12, 0.05, 0.10]),
        alpha=0.7,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params_true, n_steps=80, seed=7)
    y = out["y_true"]

    ll_true = hawkes_loglik_poisson(
        world=world, kernel=kernel, mu=params_true.mu, alpha=params_true.alpha, y=y
    )
    ll_bad = hawkes_loglik_poisson(
        world=world,
        kernel=kernel,
        mu=np.full((world.n_locations,), 0.02),
        alpha=0.05,
        y=y,
    )

    assert np.isfinite(ll_true)
    assert np.isfinite(ll_bad)
    assert ll_true > ll_bad


def test_mle_parameter_recovery_reasonable() -> None:
    world = generate_random_world(n_locations=5, seed=9, lengthscale=0.4)
    kernel = discrete_exponential_kernel(n_lags=6, beta=0.7)

    alpha_true = 0.6
    mu_true = np.linspace(0.05, 0.15, world.n_locations)
    params_true = HawkesDiscreteParams(
        mu=mu_true,
        alpha=alpha_true,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params_true, n_steps=140, seed=10)
    y = out["y_true"]

    fit = fit_hawkes_mle_alpha_mu(world=world, kernel=kernel, y=y, maxiter=400)
    alpha_hat = float(fit["alpha"])
    mu_hat = np.asarray(fit["mu"], dtype=float)

    assert alpha_hat >= 0.0
    assert mu_hat.shape == mu_true.shape
    assert np.all(mu_hat >= 0.0)

    # Ballpark recovery (stochastic + finite sample; keep tolerance loose).
    assert abs(alpha_hat - alpha_true) / alpha_true < 0.5

    # mu recovery: positive correlation and similar scale.
    corr = np.corrcoef(mu_true, mu_hat)[0, 1]
    assert corr > 0.5
    assert 0.5 <= (mu_hat.mean() / mu_true.mean()) <= 1.5


def test_hawkes_intensity_matches_simulator_output_when_using_true_history() -> None:
    world = generate_random_world(n_locations=4, seed=4, lengthscale=0.6)
    kernel = discrete_exponential_kernel(n_lags=4, beta=1.1)

    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.1),
        alpha=0.8,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=60, seed=5)
    y = out["y_true"]
    lam_expected = out["intensity"]

    lam = hawkes_intensity(world=world, kernel=kernel, mu=params.mu, alpha=params.alpha, y=y)

    assert lam.shape == lam_expected.shape
    assert np.allclose(lam, lam_expected)
