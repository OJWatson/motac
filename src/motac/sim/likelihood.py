from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .hawkes import _convolved_history
from .world import World


def hawkes_intensity(
    *,
    world: World,
    kernel: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    y: np.ndarray,
) -> np.ndarray:
    """Compute conditional intensities for a discrete-time Hawkes-like model.

    Model matches :func:`motac.sim.simulate_hawkes_counts` (latent process):

        lambda(t) = mu + alpha * (mobility @ h(t))

    where h(t) is the kernel-weighted lagged history of y.

    Parameters
    ----------
    world:
        Provides mobility matrix of shape (n_locations, n_locations).
    kernel:
        Discrete kernel over lags 1..L of shape (L,).
    mu:
        Baseline intensities per location, shape (n_locations,).
    alpha:
        Global excitation scale, non-negative.
    y:
        Count series used for history, shape (n_locations, n_steps).

    Returns
    -------
    intensity:
        Array of shape (n_locations, n_steps) with lambda(:, t).
    """

    if y.ndim != 2:
        raise ValueError("y must be a 2D array (n_locations, n_steps)")
    n_locations, n_steps = y.shape
    if n_locations != world.n_locations:
        raise ValueError("y first dimension must match world.n_locations")
    if mu.shape != (n_locations,):
        raise ValueError("mu must have shape (n_locations,)")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be a non-empty 1D array")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    intensity = np.zeros((n_locations, n_steps), dtype=float)
    for t in range(n_steps):
        h = _convolved_history(y, kernel, t)
        excitation = world.mobility @ h
        lam = mu + alpha * excitation
        intensity[:, t] = np.clip(lam, 0.0, None)
    return intensity


def hawkes_loglik_poisson(
    *,
    world: World,
    kernel: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    y: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Log-likelihood for Poisson observations under the discrete Hawkes model.

    Conditional Poisson log-likelihood:

        sum_{i,t} [ y_{i,t} log lambda_{i,t} - lambda_{i,t} - log(y_{i,t}!) ]

    Notes
    -----
    - This treats `y` as the latent (or fully observed) counts.
    - For numerical safety we lower-bound lambda by `eps` inside log.
    """

    lam = hawkes_intensity(world=world, kernel=kernel, mu=mu, alpha=alpha, y=y)
    lam_safe = np.clip(lam, eps, None)

    # gammaln(y+1) = log(y!)
    ll = (y * np.log(lam_safe) - lam_safe - gammaln(y + 1.0)).sum()
    return float(ll)
