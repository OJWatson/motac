from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..model.fit import fit_road_hawkes_mle
from ..model.forecast import forecast_intensity_horizon
from ..model.metrics import mean_negative_log_likelihood


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result from a tiny train/test backtest run."""

    n_train: int
    horizon: int
    nll: float


def backtest_fit_forecast_nll(
    *,
    travel_time_s: sp.csr_matrix,
    kernel: np.ndarray,
    y: np.ndarray,
    n_train: int,
    horizon: int,
    family: str = "poisson",
    dispersion: float | None = None,
    init_alpha: float = 0.05,
    init_beta: float = 0.05,
    maxiter: int = 250,
) -> BacktestResult:
    """Fit on a training window then score NLL on a held-out horizon.

    This is intentionally tiny (toy-data sized): it supports M5 by providing a
    single helper that wires together fit -> forecast -> scoring.

    Parameters
    ----------
    travel_time_s:
        Sparse travel-time matrix in seconds.
    kernel:
        Discrete lag kernel used by the model.
    y:
        Count matrix (n_cells, n_steps).
    n_train:
        Number of steps to use for training.
    horizon:
        Number of held-out steps to score.

    Returns
    -------
    BacktestResult with mean negative log likelihood on the held-out window.
    """

    y = np.asarray(y, dtype=int)
    if y.ndim != 2:
        raise ValueError("y must be a 2D array (n_cells, n_steps)")
    n_steps = int(y.shape[1])

    if n_train <= 1:
        raise ValueError("n_train must be > 1")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if n_train + horizon > n_steps:
        raise ValueError("n_train + horizon must be <= n_steps")

    y_train = y[:, :n_train]
    y_test = y[:, n_train : n_train + horizon]

    fit = fit_road_hawkes_mle(
        travel_time_s=travel_time_s,
        kernel=np.asarray(kernel, dtype=float),
        y=y_train,
        family=family,
        init_alpha=float(init_alpha),
        init_beta=float(init_beta),
        maxiter=int(maxiter),
    )

    lam = forecast_intensity_horizon(
        travel_time_s=travel_time_s,
        mu=np.asarray(fit["mu"], dtype=float),
        alpha=float(fit["alpha"]),
        beta=float(fit["beta"]),
        kernel=np.asarray(kernel, dtype=float),
        y_history=y_train,
        horizon=int(horizon),
    )

    nll = mean_negative_log_likelihood(
        y=y_test,
        mean=lam,
        family=family,
        dispersion=dispersion,
    )

    return BacktestResult(n_train=int(n_train), horizon=int(horizon), nll=float(nll))
