from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.inference.sparse_neighbour_ops import convolved_history_last, csr_from_scipy, csr_matvec


def test_convolved_history_last_matches_manual():
    rng = np.random.default_rng(0)
    y = rng.poisson(2.0, size=(5, 7)).astype(float)
    kernel = np.array([0.5, 0.25, 0.125])

    got = convolved_history_last(y=y, kernel=kernel)

    # Manual: last step uses lags 1..L.
    manual = (
        kernel[0] * y[:, -1]
        + kernel[1] * y[:, -2]
        + kernel[2] * y[:, -3]
    )
    assert got.shape == (5,)
    assert np.allclose(got, manual)


def test_csr_matvec_matches_scipy():
    rng = np.random.default_rng(1)
    n = 10

    # Random sparse matrix with a fixed sparsity.
    A = sp.random(n, n, density=0.2, format="csr", random_state=0)
    # Ensure the diagonal is present so edge cases are covered.
    A = (A + sp.eye(n, format="csr")).tocsr()

    x = rng.normal(size=(n,))

    csr = csr_from_scipy(A)
    got = np.asarray(csr_matvec(csr=csr, x=x), dtype=float)
    want = np.asarray(A @ x, dtype=float)

    assert got.shape == (n,)
    assert np.allclose(got, want)


def test_csr_matvec_shape_errors():
    A = (sp.eye(3, format="csr") + sp.csr_matrix(np.ones((3, 3)))).tocsr()
    csr = csr_from_scipy(A)

    try:
        csr_matvec(csr=csr, x=np.ones((3, 1)))
    except ValueError as e:
        assert "1D" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    try:
        csr_matvec(csr=csr, x=np.ones((4,)))
    except ValueError as e:
        assert "shape" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
