from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .fit import fit_road_hawkes_mle
from .forecast import forecast_intensity_horizon


def fit_forecast_road_hawkes_mle(
    *,
    travel_time_s: sp.csr_matrix,
    kernel: np.ndarray,
    y: np.ndarray,
    horizon: int,
    family: str = "poisson",
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    init_beta: float = 1e-3,
    init_dispersion: float = 10.0,
    maxiter: int = 600,
) -> dict[str, object]:
    """Convenience workflow: fit -> deterministic intensity forecast.

    This is the M3.2 glue-layer for the parametric road-kernel Hawkes model.

    Steps
    -----
    1) Fit (mu, alpha, beta) (and optionally dispersion) by MLE via
       :func:`motac.model.fit.fit_road_hawkes_mle`.
    2) Forecast the next `horizon` intensities via
       :func:`motac.model.forecast.forecast_intensity_horizon`.

    Parameters
    ----------
    travel_time_s:
        CSR travel-time neighbourhood matrix.
    kernel:
        Discrete lag kernel.
    y:
        Observed/simulated count matrix (n_cells, n_steps).
    horizon:
        Forecast horizon (>= 1).

    Returns
    -------
    dict with keys:
      - fit: dict returned by `fit_road_hawkes_mle`
      - lam_forecast: ndarray (n_cells, horizon)
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if y.ndim != 2:
        raise ValueError("y must be 2D (n_cells, n_steps)")

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    fit = fit_road_hawkes_mle(
        travel_time_s=travel_time_s,
        kernel=np.asarray(kernel, dtype=float),
        y=np.asarray(y),
        family=family,
        init_mu=init_mu,
        init_alpha=init_alpha,
        init_beta=init_beta,
        init_dispersion=init_dispersion,
        maxiter=maxiter,
    )

    mu_hat = np.asarray(fit["mu"], dtype=float)
    alpha_hat = float(fit["alpha"])
    beta_hat = float(fit["beta"])

    lam_forecast = forecast_intensity_horizon(
        travel_time_s=travel_time_s,
        mu=mu_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        kernel=np.asarray(kernel, dtype=float),
        y_history=np.asarray(y),
        horizon=int(horizon),
    )

    return {
        "fit": fit,
        "lam_forecast": lam_forecast,
    }
