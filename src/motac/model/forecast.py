from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .predict import predict_intensity_next_step


def forecast_intensity_horizon(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y_history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Deterministic intensity forecast for multiple steps ahead.

    This iterates next-step intensity prediction using a mean-field update:
    the predicted intensity at each step is appended to the history as the
    expected count for subsequent steps.

    Parameters
    ----------
    y_history:
        Past counts (n_cells, n_steps_history).
    horizon:
        Number of steps to forecast (>= 1).

    Returns
    -------
    lam_forecast:
        Array of shape (n_cells, horizon).
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    y_hist = np.asarray(y_history, dtype=float)
    if y_hist.ndim != 2:
        raise ValueError("y_history must be 2D")

    n_cells = int(y_hist.shape[0])
    out = np.zeros((n_cells, int(horizon)), dtype=float)

    for k in range(int(horizon)):
        lam_next = predict_intensity_next_step(
            travel_time_s=travel_time_s,
            mu=mu,
            alpha=alpha,
            beta=beta,
            kernel=kernel,
            y_history=y_hist,
        )
        out[:, k] = lam_next

        # Mean-field recursion: treat the predicted intensity as the expected
        # count for the next step.
        y_hist = np.concatenate([y_hist, lam_next.reshape(-1, 1)], axis=1)

    return out
