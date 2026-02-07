from __future__ import annotations

import numpy as np

from .hawkes import _convolved_history
from .world import World


def fit_hawkes_alpha_mu(
    *,
    world: World,
    kernel: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-6,
) -> dict[str, np.ndarray | float]:
    """Fit mu (per-location) and a global alpha using ridge regression.

    This is a deliberately simple parameter-recovery routine used for
    simulator sanity checks. It treats the Poisson observations as roughly
    Gaussian and regresses counts onto the history term.

    Parameters
    ----------
    world:
        Provides the mobility matrix.
    kernel:
        Discrete lag kernel of shape (n_lags,).
    y:
        Observed or latent counts of shape (n_locations, n_steps).
    ridge:
        Ridge penalty added to the normal equations.

    Returns
    -------
    dict with keys "mu" (n_locations,) and "alpha" (float).
    """

    n_locations, n_steps = y.shape
    if n_locations != world.n_locations:
        raise ValueError("y first dimension must match world.n_locations")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be a non-empty 1D array")

    # Build a design matrix per location with a single history regressor.
    # y_i(t) ≈ mu_i + alpha * x_i(t)
    X = np.zeros((n_locations, n_steps), dtype=float)
    for t in range(n_steps):
        h = _convolved_history(y, kernel, t)
        X[:, t] = world.mobility @ h

    # Estimate mu_i as intercept after accounting for alpha via pooled regression.
    # First, estimate alpha from demeaned data to remove intercepts.
    y_mean = y.mean(axis=1, keepdims=True)
    x_mean = X.mean(axis=1, keepdims=True)
    y_dm = (y - y_mean).reshape(-1)
    x_dm = (X - x_mean).reshape(-1)

    denom = float(x_dm @ x_dm + ridge)
    alpha_hat = float((x_dm @ y_dm) / denom) if denom > 0 else 0.0
    alpha_hat = float(np.clip(alpha_hat, 0.0, None))

    mu_hat = (y_mean.reshape(-1) - alpha_hat * x_mean.reshape(-1)).astype(float)
    mu_hat = np.clip(mu_hat, 0.0, None)

    return {"mu": mu_hat, "alpha": alpha_hat}
