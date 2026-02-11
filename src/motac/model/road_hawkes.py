from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def exp_travel_time_kernel(*, travel_time_s: sp.csr_matrix, beta: float) -> sp.csr_matrix:
    """Compute a sparse exponential kernel W(d) = exp(-beta * d) on travel times.

    Parameters
    ----------
    travel_time_s:
        CSR matrix of travel times in seconds.
    beta:
        Positive decay rate (1/seconds).

    Returns
    -------
    W:
        CSR matrix with same sparsity pattern as travel_time_s.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    data = np.asarray(travel_time_s.data, dtype=float)
    w_data = np.exp(-float(beta) * data)
    W = sp.csr_matrix(
        (w_data, travel_time_s.indices, travel_time_s.indptr),
        shape=travel_time_s.shape,
    )

    # Ensure the diagonal is present (self influence). Some sparse constructors
    # drop explicit zeros, so we enforce W[i,i]=1.
    W = W.tolil(copy=False)
    W.setdiag(1.0)
    return W.tocsr()


def convolved_history_last(
    *,
    y: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Compute the kernel-weighted history term h(t) for the next-step forecast.

    For a history y[:, :T], returns h(T) = sum_{l=1..L} kernel[l-1] * y[:, T-l].
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be 1D and non-empty")

    n, t = y.shape
    lags = int(kernel.size)
    start = max(0, t - lags)
    window = np.asarray(y[:, start:t], dtype=float)
    if window.size == 0:
        return np.zeros((n,), dtype=float)

    effective = t - start
    k = np.asarray(kernel[:effective], dtype=float)
    return window[:, ::-1] @ k


def predict_intensity_one_step_road(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y_history: np.ndarray,
) -> np.ndarray:
    """One-step-ahead intensity forecast using sparse road-constrained neighbours.

    Model:
      h(t) = sum_l kernel[l-1] * y(t-l)
      W = exp(-beta * d_travel)
      lambda(t) = mu + alpha * (W @ h(t))

    Parameters
    ----------
    travel_time_s:
        CSR travel-time matrix (seconds) between cells.
    mu:
        Baseline per cell (n_cells,).
    alpha:
        Non-negative excitation scale.
    beta:
        Positive travel-time decay rate.
    kernel:
        Discrete lag kernel.
    y_history:
        Past counts (n_cells, n_steps_history).

    Returns
    -------
    lam_next:
        Intensities for next step (n_cells,).
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    n_cells = int(y_history.shape[0])
    if mu.shape != (n_cells,):
        raise ValueError("mu must have shape (n_cells,)")

    h = convolved_history_last(y=y_history, kernel=kernel)
    W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)
    excitation = W @ h
    lam = np.asarray(mu, dtype=float) + float(alpha) * np.asarray(excitation, dtype=float)
    return np.clip(lam, 0.0, None)
