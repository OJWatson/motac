from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.model import simulate_road_hawkes_counts


def test_simulate_road_hawkes_counts_poisson_reproducible() -> None:
    tt = sp.csr_matrix(np.array([[0.0, 10.0], [10.0, 0.0]]))
    kernel = np.array([1.0, 0.5])

    y1 = simulate_road_hawkes_counts(
        travel_time_s=tt,
        mu=np.array([0.2, 0.3]),
        alpha=0.4,
        beta=1e-2,
        kernel=kernel,
        T=25,
        seed=123,
        family="poisson",
    )
    y2 = simulate_road_hawkes_counts(
        travel_time_s=tt,
        mu=np.array([0.2, 0.3]),
        alpha=0.4,
        beta=1e-2,
        kernel=kernel,
        T=25,
        seed=123,
        family="poisson",
    )

    assert y1.shape == (2, 25)
    assert y1.dtype == int
    assert np.array_equal(y1, y2)


def test_simulate_road_hawkes_counts_negbin_smoke() -> None:
    tt = sp.csr_matrix(np.array([[0.0, 5.0], [5.0, 0.0]]))
    kernel = np.array([1.0])

    y = simulate_road_hawkes_counts(
        travel_time_s=tt,
        mu=np.array([0.05, 0.07]),
        alpha=0.9,
        beta=5e-2,
        kernel=kernel,
        T=40,
        seed=0,
        family="negbin",
        dispersion=8.0,
    )

    assert y.shape == (2, 40)
    assert np.all(y >= 0)
