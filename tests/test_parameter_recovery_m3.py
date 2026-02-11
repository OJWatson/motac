from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.fit import fit_road_hawkes_mle
from motac.model.road_hawkes import convolved_history_last, exp_travel_time_kernel


def _simulate_poisson_road_counts(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate counts sequentially under the road-constrained Poisson model."""

    n_cells = int(mu.shape[0])
    W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)

    y = np.zeros((n_cells, int(n_steps)), dtype=int)
    for t in range(int(n_steps)):
        h = convolved_history_last(y=y[:, :t], kernel=kernel)
        lam = mu + float(alpha) * (W @ h)
        lam = np.clip(np.asarray(lam, dtype=float), 0.0, None)
        y[:, t] = rng.poisson(lam=lam)

    return y


def test_parameter_recovery_road_poisson_multiseed() -> None:
    # Tiny 3-cell substrate represented purely by a sparse travel-time matrix.
    # Keep this offline and fast for CI.
    d = sp.csr_matrix(
        np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 0.0, 12.0],
                [20.0, 12.0, 0.0],
            ]
        )
    )

    mu_true = np.array([0.20, 0.25, 0.15])
    alpha_true = 0.35
    beta_true = 0.08
    kernel = np.array([1.0, 0.5])  # 2 lags

    # Multi-seed to reduce flakiness.
    seeds = [0, 1, 2]

    alpha_hats: list[float] = []
    beta_hats: list[float] = []
    mu_mean_hats: list[float] = []

    for s in seeds:
        rng = np.random.default_rng(s)
        y = _simulate_poisson_road_counts(
            travel_time_s=d,
            mu=mu_true,
            alpha=alpha_true,
            beta=beta_true,
            kernel=kernel,
            n_steps=80,
            rng=rng,
        )

        fit = fit_road_hawkes_mle(
            travel_time_s=d,
            kernel=kernel,
            y=y,
            family="poisson",
            init_alpha=0.1,
            init_beta=0.02,
            maxiter=300,
        )

        alpha_hat = float(fit["alpha"])
        beta_hat = float(fit["beta"])
        mu_hat = np.asarray(fit["mu"], dtype=float)

        assert np.all(np.isfinite(mu_hat))
        assert np.isfinite(alpha_hat)
        assert np.isfinite(beta_hat)
        assert alpha_hat >= 0.0
        assert beta_hat > 0.0

        alpha_hats.append(alpha_hat)
        beta_hats.append(beta_hat)
        mu_mean_hats.append(float(mu_hat.mean()))

    # Coarse recovery tolerances (stability > precision for CI).
    alpha_med = float(np.median(alpha_hats))
    beta_med = float(np.median(beta_hats))
    mu_mean_med = float(np.median(mu_mean_hats))

    assert 0.5 * alpha_true <= alpha_med <= 2.0 * alpha_true
    assert 0.5 * beta_true <= beta_med <= 2.0 * beta_true

    mu_mean_true = float(mu_true.mean())
    assert 0.7 * mu_mean_true <= mu_mean_med <= 1.3 * mu_mean_true
