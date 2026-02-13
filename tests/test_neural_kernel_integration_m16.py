import numpy as np
import scipy.sparse as sp


def test_predict_intensity_one_step_road_accepts_kernel_fn() -> None:
    from motac.model.road_hawkes import predict_intensity_one_step_road

    # 2-cell toy graph with symmetric nonzero travel times.
    tt = sp.csr_matrix(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float))

    # Simple 1-lag history: h(T) = y(T-1)
    y_hist = np.array([[3.0], [5.0]], dtype=float)
    kernel = np.array([1.0], dtype=float)

    mu = np.array([0.5, 0.5], dtype=float)
    alpha = 2.0

    def zero_offdiag(d: np.ndarray) -> np.ndarray:
        # Return zeros for all edges; adapter will still enforce diag=1.
        return np.zeros_like(d, dtype=float)

    lam = predict_intensity_one_step_road(
        travel_time_s=tt,
        mu=mu,
        alpha=alpha,
        beta=123.0,
        kernel=kernel,
        y_history=y_hist,
        kernel_fn=zero_offdiag,
    )

    # With off-diagonal weights forced to 0 and diag=1, excitation is just h.
    expected = mu + alpha * np.array([3.0, 5.0], dtype=float)
    np.testing.assert_allclose(lam, expected)


def test_road_intensity_matrix_accepts_kernel_fn() -> None:
    from motac.model.likelihood import road_intensity_matrix

    tt = sp.csr_matrix(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float))

    mu = np.array([0.5, 0.5], dtype=float)
    alpha = 2.0
    kernel = np.array([1.0], dtype=float)

    # Two time steps: at t=0, history empty -> lambda=mu.
    # At t=1, h=y[:,0].
    y = np.array([[3.0, 0.0], [5.0, 0.0]], dtype=float)

    def zero_offdiag(d: np.ndarray) -> np.ndarray:
        return np.zeros_like(d, dtype=float)

    lam = road_intensity_matrix(
        travel_time_s=tt,
        mu=mu,
        alpha=alpha,
        beta=123.0,
        kernel=kernel,
        y=y,
        kernel_fn=zero_offdiag,
    )

    expected = np.stack(
        [
            mu,
            mu + alpha * np.array([3.0, 5.0], dtype=float),
        ],
        axis=1,
    )
    np.testing.assert_allclose(lam, expected)
