from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def test_travel_time_kernel_from_fn_builds_sparse_weights() -> None:
    from motac.model.neural_kernels import ExpDecayKernel
    from motac.model.road_hawkes import travel_time_kernel_from_fn

    # Small 3x3 travel-time graph with a few edges (seconds).
    tt = sp.csr_matrix(
        (
            np.asarray([2.0, 5.0, 1.0], dtype=np.float64),
            np.asarray([1, 2, 0], dtype=np.int32),
            np.asarray([0, 2, 3, 3], dtype=np.int32),
        ),
        shape=(3, 3),
    )

    kernel = ExpDecayKernel(lengthscale=2.0)
    W = travel_time_kernel_from_fn(travel_time_s=tt, kernel_fn=kernel)

    assert sp.isspmatrix_csr(W)
    assert W.shape == tt.shape

    # Off-diagonal weights are kernel(d).
    assert np.isclose(W[0, 1], np.exp(-2.0 / 2.0))
    assert np.isclose(W[0, 2], np.exp(-5.0 / 2.0))
    assert np.isclose(W[1, 0], np.exp(-1.0 / 2.0))

    # Diagonal is forced to 1.0 for self-influence.
    assert np.isclose(W[0, 0], 1.0)
    assert np.isclose(W[1, 1], 1.0)
    assert np.isclose(W[2, 2], 1.0)

    # Basic invariant: nonnegative weights.
    assert np.all(np.asarray(W.data) >= 0.0)
