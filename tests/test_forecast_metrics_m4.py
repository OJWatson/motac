from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.forecast import forecast_intensity_horizon
from motac.model.metrics import mean_negative_log_likelihood


def test_forecast_intensity_horizon_shapes_and_determinism() -> None:
    d = sp.csr_matrix(
        np.array(
            [
                [0.0, 10.0],
                [10.0, 0.0],
            ]
        )
    )

    mu = np.array([0.2, 0.1])
    alpha = 0.2
    beta = 0.1
    kernel = np.array([0.6, 0.2])

    y_hist = np.array([[0, 1], [1, 0]], dtype=int)

    lam1 = forecast_intensity_horizon(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y_hist,
        horizon=3,
    )
    lam2 = forecast_intensity_horizon(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y_hist,
        horizon=3,
    )

    assert lam1.shape == (2, 3)
    assert np.all(np.isfinite(lam1))
    assert np.all(lam1 >= 0.0)
    assert np.allclose(lam1, lam2)


def test_mean_negative_log_likelihood_poisson_smoke() -> None:
    y = np.array([[0, 1, 0], [2, 0, 1]], dtype=int)
    mean = np.full_like(y, 1.0, dtype=float)
    nll = mean_negative_log_likelihood(y=y, mean=mean, family="poisson")
    assert np.isfinite(nll)
    assert nll >= 0.0
