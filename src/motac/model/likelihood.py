from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .neural_kernels import KernelFn
from .road_hawkes import (
    convolved_history_last,
    exp_travel_time_kernel,
    travel_time_kernel_from_fn,
)


def negbin_logpmf(*, y: np.ndarray, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Negative binomial log PMF parameterised by mean and dispersion.

    We use the common NB2 parameterisation:

        Var[Y] = mean + mean^2 / dispersion,

    where `dispersion` > 0 (larger -> closer to Poisson).

    Parameters
    ----------
    y:
        Counts.
    mean:
        Mean parameter (same shape as y).
    dispersion:
        Positive dispersion parameter.

    Returns
    -------
    logpmf:
        Array of log PMF values with same shape as y.
    """

    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    y = np.asarray(y, dtype=float)
    m = np.asarray(mean, dtype=float)
    if y.shape != m.shape:
        raise ValueError("y and mean must have the same shape")

    # NB as Gamma-Poisson mixture: shape=k, scale=mean/k.
    k = float(dispersion)
    # log Gamma(y+k) - log Gamma(k) - log y!
    log_coeff = gammaln(y + k) - gammaln(k) - gammaln(y + 1.0)
    log_p = k * (np.log(k) - np.log(k + m))
    log_q = y * (np.log(m) - np.log(k + m))
    return log_coeff + log_p + log_q


def poisson_logpmf(*, y: np.ndarray, mean: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Poisson log PMF with safe log."""

    y = np.asarray(y, dtype=float)
    m = np.asarray(mean, dtype=float)
    if y.shape != m.shape:
        raise ValueError("y and mean must have the same shape")

    m_safe = np.clip(m, eps, None)
    return y * np.log(m_safe) - m_safe - gammaln(y + 1.0)


def road_intensity_matrix(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y: np.ndarray,
    kernel_fn: KernelFn | None = None,
    validate_kernel: bool = True,
) -> np.ndarray:
    """Compute intensities lambda[:, t] for all t given a count series y.

    This uses the sparse road-constrained kernel W(d_travel)=exp(-beta*d) (or
    W(d_travel)=kernel_fn(d_travel)) and the lag kernel to build a Hawkes-like
    recursion.

    Parameters
    ----------
    y:
        Count matrix of shape (n_cells, n_steps).

    Returns
    -------
    intensity:
        Array of shape (n_cells, n_steps).
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D")
    n_cells, n_steps = y.shape

    if mu.shape != (n_cells,):
        raise ValueError("mu must have shape (n_cells,)")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    if kernel_fn is None:
        W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)
    else:
        W = travel_time_kernel_from_fn(
            travel_time_s=travel_time_s,
            kernel_fn=kernel_fn,
            validate=validate_kernel,
        )

    intensity = np.zeros((n_cells, n_steps), dtype=float)

    # For each t, compute h(t) based on y[:, :t], then lambda(t).
    for t in range(n_steps):
        h = convolved_history_last(y=y[:, :t], kernel=kernel)
        excitation = W @ h
        lam = np.asarray(mu, dtype=float) + float(alpha) * np.asarray(excitation, dtype=float)
        intensity[:, t] = np.clip(lam, 0.0, None)

    return intensity


def road_loglik(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    y: np.ndarray,
    family: str = "poisson",
    dispersion: float | None = None,
    kernel_fn: KernelFn | None = None,
    validate_kernel: bool = True,
) -> float:
    """Log-likelihood for road-constrained count model under Poisson or NegBin."""

    lam = road_intensity_matrix(
        travel_time_s=travel_time_s,
        mu=mu,
        alpha=alpha,
        beta=beta,
        kernel=kernel,
        y=y,
        kernel_fn=kernel_fn,
        validate_kernel=validate_kernel,
    )

    if family == "poisson":
        ll = poisson_logpmf(y=y, mean=lam).sum()
        return float(ll)

    if family == "negbin":
        if dispersion is None:
            raise ValueError("dispersion must be provided for negbin")
        ll = negbin_logpmf(y=y, mean=lam, dispersion=float(dispersion)).sum()
        return float(ll)

    raise ValueError("family must be one of {'poisson','negbin'}")
