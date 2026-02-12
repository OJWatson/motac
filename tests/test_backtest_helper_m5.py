from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.eval import backtest_fit_forecast_nll
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


def test_backtest_fit_forecast_nll_toy_poisson() -> None:
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

    res = backtest_fit_forecast_nll(
        travel_time_s=d,
        kernel=kernel,
        y=y,
        n_train=40,
        horizon=5,
        family="poisson",
        maxiter=200,
    )

    assert res.n_train == 40
    assert res.horizon == 5
    assert np.isfinite(res.nll)
    assert res.nll >= 0.0
