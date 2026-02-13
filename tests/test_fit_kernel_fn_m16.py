import numpy as np
import pytest
import scipy.sparse as sp

from motac.model.fit import fit_road_hawkes_mle


def test_fit_road_hawkes_mle_rejects_invalid_kernel_fn() -> None:
    travel_time_s = sp.csr_matrix(np.array([[0.0, 10.0], [10.0, 0.0]]))
    y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kernel = np.array([1.0])

    def bad_kernel_fn(d: np.ndarray) -> np.ndarray:
        return -np.ones_like(d, dtype=float)

    with pytest.raises(ValueError, match="nonnegative"):
        fit_road_hawkes_mle(
            travel_time_s=travel_time_s,
            kernel=kernel,
            y=y,
            kernel_fn=bad_kernel_fn,
            maxiter=2,
        )
