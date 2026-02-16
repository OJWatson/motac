from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from motac.substrate.types import Substrate


def _exp_travel_time_weights(*, travel_time_s: sp.csr_matrix, beta: float) -> sp.csr_matrix:
    """Sparse exponential weights W(d)=exp(-beta*d) with ensured diagonal.

    Parameters
    ----------
    travel_time_s:
        CSR matrix of nonnegative travel times in seconds.
    beta:
        Positive decay rate (1/seconds).
    """

    if beta <= 0:
        raise ValueError("beta must be positive")

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    data = np.asarray(travel_time_s.data, dtype=float)
    if np.any(data < 0):
        raise ValueError("travel_time_s must be nonnegative")

    w_data = np.exp(-float(beta) * data)
    W = sp.csr_matrix(
        (w_data, travel_time_s.indices, travel_time_s.indptr),
        shape=travel_time_s.shape,
    )

    # Ensure diagonal is present; some constructors drop explicit zeros.
    W = W.tolil(copy=False)
    W.setdiag(1.0)
    return W.tocsr()


def _convolved_history_last(*, y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute h(T)=sum_{l=1..L} kernel[l-1] * y[:,T-l] for y[:, :T]."""

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


@dataclass(frozen=True, slots=True)
class SubstrateHawkesParams:
    """Discrete-time Hawkes-like params on a road-constrained substrate."""

    mu: np.ndarray  # (n_cells,)
    alpha: float
    beta: float
    kernel: np.ndarray  # (n_lags,)

    def __post_init__(self) -> None:
        if self.mu.ndim != 1:
            raise ValueError("mu must be 1D")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.kernel.ndim != 1 or self.kernel.size == 0:
            raise ValueError("kernel must be 1D and non-empty")


def simulate_substrate_hawkes_counts(
    *,
    substrate: Substrate,
    params: SubstrateHawkesParams,
    n_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate latent counts on a substrate using sparse travel-time neighbours.

    Model
    -----
      h(t) = sum_{l=1..L} kernel[l-1] * y(t-l)
      W(d) = exp(-beta * d_travel)
      lambda(t) = mu + alpha * (W @ h(t))
      y(t) ~ Poisson(lambda(t))

    Returns a dict with arrays:
      - y_true: (n_cells, n_steps)
      - intensity: (n_cells, n_steps)

    Notes
    -----
    This is the simulator side of the road-constrained parametric Hawkes model
    used in `motac.model`. It is intentionally dependency-light and deterministic
    given a seed.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    tt = substrate.neighbours.travel_time_s
    if not sp.isspmatrix_csr(tt):
        tt = tt.tocsr()

    n_cells = int(tt.shape[0])
    if tt.shape != (n_cells, n_cells):
        raise ValueError("substrate.neighbours.travel_time_s must be square")
    if params.mu.shape != (n_cells,):
        raise ValueError("params.mu must have shape (n_cells,)")

    W = _exp_travel_time_weights(travel_time_s=tt, beta=float(params.beta))

    rng = np.random.default_rng(int(seed))
    y_true = np.zeros((n_cells, int(n_steps)), dtype=int)
    intensity = np.zeros((n_cells, int(n_steps)), dtype=float)

    for t in range(int(n_steps)):
        h_t = _convolved_history_last(y=y_true[:, :t], kernel=params.kernel)
        lam_t = params.mu + float(params.alpha) * (W @ h_t)
        lam_t = np.clip(np.asarray(lam_t, dtype=float), 0.0, None)
        intensity[:, t] = lam_t
        y_true[:, t] = rng.poisson(lam_t)

    return {"y_true": y_true, "intensity": intensity}
