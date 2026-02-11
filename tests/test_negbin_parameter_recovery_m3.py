from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.fit import fit_road_hawkes_mle
from motac.model.road_hawkes import convolved_history_last, exp_travel_time_kernel


def _simulate_negbin_road_counts(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    dispersion: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate counts under NB2 using a Gamma-Poisson mixture.

    For each cell/time with mean lambda and dispersion k:
      rate ~ Gamma(shape=k, scale=lambda/k)
      y ~ Poisson(rate)

    This yields Var[y] = lambda + lambda^2/k.
    """

    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    n_cells = int(mu.shape[0])
    W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)
    k = float(dispersion)

    y = np.zeros((n_cells, int(n_steps)), dtype=int)
    for t in range(int(n_steps)):
        h = convolved_history_last(y=y[:, :t], kernel=kernel)
        lam = mu + float(alpha) * (W @ h)
        lam = np.clip(np.asarray(lam, dtype=float), 1e-12, None)

        # Gamma-Poisson mixture
        rate = rng.gamma(shape=k, scale=lam / k)
        y[:, t] = rng.poisson(lam=rate)

    return y


def test_parameter_recovery_road_negbin_multiseed_smoke() -> None:
    # Tiny offline travel-time matrix.
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
    alpha_true = 0.15
    beta_true = 0.08
    disp_true = 20.0
    kernel = np.array([0.6, 0.2])

    seeds = [0, 1, 2]
    disp_hats: list[float] = []

    for s in seeds:
        rng = np.random.default_rng(s)
        y = _simulate_negbin_road_counts(
            travel_time_s=d,
            mu=mu_true,
            alpha=alpha_true,
            beta=beta_true,
            kernel=kernel,
            dispersion=disp_true,
            n_steps=120,
            rng=rng,
        )

        fit = fit_road_hawkes_mle(
            travel_time_s=d,
            kernel=kernel,
            y=y,
            family="negbin",
            init_alpha=0.1,
            init_beta=0.05,
            init_dispersion=10.0,
            maxiter=500,
        )

        disp_hat = float(fit["dispersion"])
        assert np.isfinite(disp_hat)
        assert disp_hat > 0.0
        disp_hats.append(disp_hat)

    # Coarse dispersion recovery: avoid flakiness.
    disp_med = float(np.median(disp_hats))
    assert 0.2 * disp_true <= disp_med <= 5.0 * disp_true
