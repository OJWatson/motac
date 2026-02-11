from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.road_hawkes import predict_intensity_one_step_road


def test_predict_intensity_one_step_road_toy() -> None:
    # 2 cells with asymmetric travel times.
    d = sp.csr_matrix(np.array([[0.0, 10.0], [5.0, 0.0]]))

    mu = np.array([0.1, 0.2])
    alpha = 0.5
    beta = 0.1
    kernel = np.array([1.0])

    y_hist = np.array([[2, 0], [1, 0]], dtype=int)  # last step counts are y[:,1]

    # With kernel lag 1 and history length 2, h(T)=y[:,1]=[0,0].
    lam = predict_intensity_one_step_road(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y_hist,
    )
    assert np.allclose(lam, mu)

    # If we put counts in last step, excitation should be positive.
    y_hist2 = np.array([[0, 2], [0, 1]], dtype=int)
    lam2 = predict_intensity_one_step_road(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y_hist2,
    )

    # W = exp(-beta*d)
    W = np.exp(-beta * np.array([[0.0, 10.0], [5.0, 0.0]]))
    h = np.array([2.0, 1.0])
    expected = mu + alpha * (W @ h)
    assert lam2.shape == (2,)
    assert np.all(np.isfinite(lam2))
    assert np.all(lam2 >= 0.0)
    assert np.allclose(lam2, expected)
