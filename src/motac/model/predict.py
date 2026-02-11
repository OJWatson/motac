from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .likelihood import road_intensity_matrix
from .road_hawkes import predict_intensity_one_step_road


def predict_intensity_in_sample(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Predict in-sample intensities lambda[:, t] for an observed count series."""

    return road_intensity_matrix(
        travel_time_s=travel_time_s,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y=y,
    )


def predict_intensity_next_step(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y_history: np.ndarray,
) -> np.ndarray:
    """Predict next-step intensities given a history y[:, :T]."""

    return predict_intensity_one_step_road(
        travel_time_s=travel_time_s,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y_history,
    )
