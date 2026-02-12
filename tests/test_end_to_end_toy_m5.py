from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.fit import fit_road_hawkes_mle
from motac.model.forecast import forecast_intensity_horizon
from motac.model.metrics import mean_negative_log_likelihood
from motac.model.road_hawkes import convolved_history_last, exp_travel_time_kernel


def _simulate_poisson(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = int(mu.shape[0])
    W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)

    y = np.zeros((n_cells, int(n_steps)), dtype=int)
    for t in range(int(n_steps)):
        h = convolved_history_last(y=y[:, :t], kernel=kernel)
        lam = np.clip(mu + float(alpha) * (W @ h), 1e-12, None)
        y[:, t] = rng.poisson(lam=np.asarray(lam, dtype=float))

    return y


def test_fit_forecast_score_toy_poisson() -> None:
    d = sp.csr_matrix(np.array([[0.0, 10.0], [10.0, 0.0]]))

    mu_true = np.array([0.15, 0.10])
    alpha_true = 0.12
    beta_true = 0.08
    kernel = np.array([0.6, 0.2])

    y = _simulate_poisson(
        travel_time_s=d,
        mu=mu_true,
        alpha=alpha_true,
        beta=beta_true,
        kernel=kernel,
        n_steps=50,
        seed=0,
    )

    n_train = 40
    horizon = 5
    y_train = y[:, :n_train]
    y_test = y[:, n_train : n_train + horizon]

    fit = fit_road_hawkes_mle(
        travel_time_s=d,
        kernel=kernel,
        y=y_train,
        family="poisson",
        init_alpha=0.05,
        init_beta=0.05,
        maxiter=250,
    )

    mu_hat = np.asarray(fit["mu"], dtype=float)
    alpha_hat = float(fit["alpha"])
    beta_hat = float(fit["beta"])

    lam_hat = forecast_intensity_horizon(
        travel_time_s=d,
        mu=mu_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        kernel=kernel,
        y_history=y_train,
        horizon=horizon,
    )

    assert lam_hat.shape == y_test.shape
    assert np.all(np.isfinite(lam_hat))
    assert np.all(lam_hat >= 0.0)

    nll = mean_negative_log_likelihood(y=y_test, mean=lam_hat, family="poisson")
    assert np.isfinite(nll)
    assert nll >= 0.0


def test_mean_negative_log_likelihood_negbin_smoke() -> None:
    y = np.array([[0, 1, 0], [2, 0, 1]], dtype=int)
    mean = np.full_like(y, 1.0, dtype=float)

    nll = mean_negative_log_likelihood(y=y, mean=mean, family="negbin", dispersion=10.0)
    assert np.isfinite(nll)
    assert nll >= 0.0
