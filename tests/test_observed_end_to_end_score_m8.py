from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from motac.sim import (
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu_observed_poisson_approx,
    generate_random_world,
    sample_hawkes_observed_predictive_paths_poisson_approx,
)


def _poisson_nll(*, y: np.ndarray, lam: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log-likelihood under independent Poisson(y | lam)."""

    lam_safe = np.clip(lam, eps, None)
    ll = (y * np.log(lam_safe) - lam_safe - gammaln(y + 1.0)).sum()
    return float(-ll)


def test_observed_end_to_end_fit_forecast_score_toy() -> None:
    """Minimal observed-only workflow test: fit -> predictive sample -> score.

    This is intended to be CI-safe (small sizes, deterministic RNG).
    """

    world = generate_random_world(n_locations=3, seed=0, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=3, beta=1.0)

    # Toy observed series with a small pulse.
    y_obs = np.zeros((world.n_locations, 20), dtype=int)
    y_obs[:, 5] = 2
    y_obs[:, 12] = 1

    t_train = 15
    y_train = y_obs[:, :t_train]
    y_test = y_obs[:, t_train:]

    p_detect = 0.7
    false_rate = 0.2

    fit = fit_hawkes_mle_alpha_mu_observed_poisson_approx(
        world=world,
        kernel=kernel,
        y_true_for_history=y_train,
        y_obs=y_train,
        p_detect=p_detect,
        false_rate=false_rate,
        init_alpha=0.1,
        maxiter=50,
    )

    mu_hat = np.asarray(fit["mu"], dtype=float)
    alpha_hat = float(fit["alpha"])

    paths = sample_hawkes_observed_predictive_paths_poisson_approx(
        world=world,
        mu=mu_hat,
        alpha=alpha_hat,
        kernel=kernel,
        y_history_for_intensity=y_train,
        horizon=y_test.shape[1],
        n_paths=20,
        seed=123,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    assert paths["y_obs"].shape == (20, world.n_locations, y_test.shape[1])
    assert paths["intensity_obs"].shape == (20, world.n_locations, y_test.shape[1])

    # Simple score: Poisson NLL using the Monte Carlo mean observed intensity.
    lam_mean = paths["intensity_obs"].mean(axis=0)
    nll = _poisson_nll(y=y_test.astype(float), lam=lam_mean)

    assert np.isfinite(nll)
    assert nll >= 0.0
