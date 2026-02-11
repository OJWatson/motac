from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model.predict import predict_intensity_in_sample, predict_intensity_next_step


def test_predict_intensity_shapes_and_consistency() -> None:
    d = sp.csr_matrix(
        np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 0.0, 12.0],
                [20.0, 12.0, 0.0],
            ]
        )
    )

    mu = np.array([0.2, 0.1, 0.15])
    alpha = 0.2
    beta = 0.08
    kernel = np.array([0.6, 0.2])

    y = np.array(
        [
            [0, 1, 0, 2],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    )

    lam = predict_intensity_in_sample(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y=y,
    )

    assert lam.shape == y.shape
    assert np.all(np.isfinite(lam))
    assert np.all(lam >= 0.0)

    # Next-step prediction should match the final column if we append a dummy step.
    lam_next = predict_intensity_next_step(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y_history=y,
    )
    assert lam_next.shape == (y.shape[0],)
    assert np.all(np.isfinite(lam_next))
    assert np.all(lam_next >= 0.0)

    lam2 = predict_intensity_in_sample(
        travel_time_s=d,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y=np.concatenate([y, np.zeros((y.shape[0], 1), dtype=int)], axis=1),
    )
    assert np.allclose(lam_next, lam2[:, -1])
