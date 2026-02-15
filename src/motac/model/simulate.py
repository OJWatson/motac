from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .neural_kernels import KernelFn
from .road_hawkes import (
    convolved_history_last,
    exp_travel_time_kernel,
    travel_time_kernel_from_fn,
)


def _sample_negbin_mean_disp(
    rng: np.random.Generator,
    mean: np.ndarray,
    dispersion: float,
) -> np.ndarray:
    """Sample Negative Binomial counts with mean/dispersion parameterisation.

    Uses NB2-style parameterisation where:
      Var[Y] = mean + mean^2 / dispersion

    Here `dispersion` is the 'size' parameter (often r or k) and must be > 0.
    """

    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    mean = np.asarray(mean, dtype=float)
    mean = np.clip(mean, 0.0, None)

    r = float(dispersion)
    # numpy negative_binomial expects integer number of failures `n` and success prob `p`.
    # It supports non-integer n via Gamma-Poisson mixture equivalence.
    # mean = n*(1-p)/p  => p = n/(n+mean)
    p = r / (r + mean + 1e-12)
    return rng.negative_binomial(r, p, size=mean.shape)


def simulate_road_hawkes_counts(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    T: int,
    seed: int = 0,
    family: str = "poisson",
    dispersion: float | None = None,
    kernel_fn: KernelFn | None = None,
    validate_kernel: bool = True,
) -> np.ndarray:
    """Simulate discrete-time road-constrained Hawkes-like counts.

    Model (counts):
      h(t) = sum_{l=1..L} kernel[l-1] * y(t-l)
      W = exp(-beta * d_travel)  (or W = kernel_fn(d_travel))
      lambda(t) = mu + alpha * (W @ h(t))
      y(t) ~ Poisson(lambda(t)) or NegBin(mean=lambda(t), dispersion)

    Parameters
    ----------
    travel_time_s:
        CSR matrix of travel times in seconds.
    mu:
        Baseline per cell of shape (n_cells,).
    alpha:
        Non-negative excitation scale.
    beta:
        Positive travel-time decay rate. Ignored when `kernel_fn` is provided.
    kernel:
        Discrete lag kernel weights (1D, non-empty).
    T:
        Number of time steps to simulate.
    seed:
        RNG seed.
    family:
        "poisson" or "negbin".
    dispersion:
        Required when family="negbin"; NB 'size' parameter.
    kernel_fn:
        Optional travel-time kernel W(d) overriding exp(-beta*d).

    Returns
    -------
    y:
        Simulated counts with shape (n_cells, T).
    """

    if T <= 0:
        raise ValueError("T must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be 1D and non-empty")

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    mu = np.asarray(mu, dtype=float)
    n_cells = int(travel_time_s.shape[0])
    if travel_time_s.shape[0] != travel_time_s.shape[1]:
        raise ValueError("travel_time_s must be square")
    if mu.shape != (n_cells,):
        raise ValueError("mu must have shape (n_cells,)")

    if kernel_fn is None:
        W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=float(beta))
    else:
        W = travel_time_kernel_from_fn(
            travel_time_s=travel_time_s,
            kernel_fn=kernel_fn,
            validate=validate_kernel,
        )

    rng = np.random.default_rng(int(seed))
    y = np.zeros((n_cells, int(T)), dtype=int)

    for t in range(int(T)):
        h_t = convolved_history_last(y=y[:, :t], kernel=kernel)
        lam_t = mu + float(alpha) * (W @ h_t)
        lam_t = np.clip(np.asarray(lam_t, dtype=float), 0.0, None)

        if family == "poisson":
            y[:, t] = rng.poisson(lam_t)
        elif family == "negbin":
            if dispersion is None:
                raise ValueError("dispersion is required when family='negbin'")
            y[:, t] = _sample_negbin_mean_disp(rng, lam_t, float(dispersion))
        else:
            raise ValueError("family must be 'poisson' or 'negbin'")

    return y
