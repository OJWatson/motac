from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .hawkes import _convolved_history, discrete_exponential_kernel
from .likelihood import hawkes_loglik_poisson, hawkes_loglik_poisson_observed
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


def _softplus(x: np.ndarray) -> np.ndarray:
    # Stable softplus.
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def fit_hawkes_mle_alpha_mu(
    *,
    world: World,
    kernel: np.ndarray,
    y: np.ndarray,
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    maxiter: int = 500,
) -> dict[str, np.ndarray | float | object]:
    """Fit (mu, alpha) by maximum likelihood under the Poisson Hawkes model.

    This is intended as the first "parametric Hawkes" vertical slice (M3):
    likelihood + fit + prediction hooks for simulator recovery.

    Parameters
    ----------
    world, kernel, y:
        See :func:`motac.sim.hawkes_loglik_poisson`.
    init_mu:
        Optional initial mu guess, shape (n_locations,). If None, uses
        per-location sample mean.
    init_alpha:
        Initial alpha guess.
    maxiter:
        Maximum iterations for the optimiser.

    Returns
    -------
    dict with keys:
      - mu: np.ndarray (n_locations,)
      - alpha: float
      - loglik: float
      - result: scipy optimisation result
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D")
    n_locations, _ = y.shape
    if n_locations != world.n_locations:
        raise ValueError("y first dimension must match world.n_locations")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be a non-empty 1D array")

    y_mean = y.mean(axis=1)
    mu0 = (y_mean if init_mu is None else np.asarray(init_mu, dtype=float)).copy()
    if mu0.shape != (n_locations,):
        raise ValueError("init_mu must have shape (n_locations,)")
    mu0 = np.clip(mu0, 1e-6, None)
    alpha0 = float(max(init_alpha, 0.0))

    # Unconstrained parameterisation:
    #   mu = softplus(theta_mu) + eps, alpha = softplus(theta_alpha)
    theta0 = np.concatenate([
        np.log(np.expm1(mu0) + 1e-6),
        np.array([np.log(np.expm1(alpha0) + 1e-6)]),
    ])

    def objective(theta: np.ndarray) -> float:
        theta_mu = theta[:n_locations]
        theta_alpha = theta[n_locations]
        mu = _softplus(theta_mu) + 1e-12
        alpha = float(_softplus(np.array([theta_alpha]))[0])
        return -hawkes_loglik_poisson(
            world=world, kernel=kernel, mu=mu, alpha=alpha, y=y
        )

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    theta_hat = np.asarray(res.x, dtype=float)
    mu_hat = _softplus(theta_hat[:n_locations]) + 1e-12
    alpha_hat = float(_softplus(np.array([theta_hat[n_locations]]))[0])
    ll = hawkes_loglik_poisson(
        world=world, kernel=kernel, mu=mu_hat, alpha=alpha_hat, y=y
    )

    return {
        "mu": mu_hat,
        "alpha": alpha_hat,
        "loglik": float(ll),
        "result": res,
    }


def fit_hawkes_mle_alpha_mu_beta(
    *,
    world: World,
    n_lags: int,
    y: np.ndarray,
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    init_beta: float = 1.0,
    maxiter: int = 800,
) -> dict[str, np.ndarray | float | object]:
    r"""Fit (mu, alpha, beta) by MLE with a 1-parameter exponential kernel.

    Kernel parameterisation
    -----------------------
    We fix the lag length L = n_lags and use a normalized exponential kernel
    over lags 1..L:

        g_l \propto exp(-beta * (l-1))

    with beta > 0 enforced via a softplus transform.

    Returns
    -------
    dict with keys:
      - mu: np.ndarray (n_locations,)
      - alpha: float
      - beta: float
      - kernel: np.ndarray (n_lags,)
      - loglik: float
      - loglik_init: float
      - result: scipy optimisation result
    """

    if n_lags <= 0:
        raise ValueError("n_lags must be positive")
    if y.ndim != 2:
        raise ValueError("y must be 2D")
    n_locations, _ = y.shape
    if n_locations != world.n_locations:
        raise ValueError("y first dimension must match world.n_locations")

    y_mean = y.mean(axis=1)
    mu0 = (y_mean if init_mu is None else np.asarray(init_mu, dtype=float)).copy()
    if mu0.shape != (n_locations,):
        raise ValueError("init_mu must have shape (n_locations,)")
    mu0 = np.clip(mu0, 1e-6, None)

    alpha0 = float(max(init_alpha, 0.0))
    beta0 = float(max(init_beta, 0.0))

    # Unconstrained parameterisation:
    #   mu = softplus(theta_mu) + eps
    #   alpha = softplus(theta_alpha)
    #   beta = softplus(theta_beta) + eps
    theta0 = np.concatenate(
        [
            np.log(np.expm1(mu0) + 1e-6),
            np.array(
                [
                    np.log(np.expm1(alpha0) + 1e-6),
                    np.log(np.expm1(beta0) + 1e-6),
                ]
            ),
        ]
    )

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
        theta_mu = theta[:n_locations]
        theta_alpha = theta[n_locations]
        theta_beta = theta[n_locations + 1]

        mu = _softplus(theta_mu) + 1e-12
        alpha = float(_softplus(np.array([theta_alpha]))[0])
        beta = float(_softplus(np.array([theta_beta]))[0] + 1e-12)
        kernel = discrete_exponential_kernel(n_lags=n_lags, beta=beta, normalize=True)
        return mu, alpha, beta, kernel

    mu_init, alpha_init, beta_init, kernel_init = unpack(theta0)
    ll_init = hawkes_loglik_poisson(
        world=world,
        kernel=kernel_init,
        mu=mu_init,
        alpha=alpha_init,
        y=y,
    )

    def objective(theta: np.ndarray) -> float:
        mu, alpha, _beta, kernel = unpack(theta)
        return -hawkes_loglik_poisson(world=world, kernel=kernel, mu=mu, alpha=alpha, y=y)

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    theta_hat = np.asarray(res.x, dtype=float)
    mu_hat, alpha_hat, beta_hat, kernel_hat = unpack(theta_hat)
    ll = hawkes_loglik_poisson(
        world=world,
        kernel=kernel_hat,
        mu=mu_hat,
        alpha=alpha_hat,
        y=y,
    )

    return {
        "mu": mu_hat,
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "kernel": kernel_hat,
        "loglik": float(ll),
        "loglik_init": float(ll_init),
        "result": res,
    }


def fit_hawkes_mle_alpha_mu_observed_poisson_approx(
    *,
    world: World,
    kernel: np.ndarray,
    y_true_for_history: np.ndarray,
    y_obs: np.ndarray,
    p_detect: float,
    false_rate: float,
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    maxiter: int = 600,
) -> dict[str, np.ndarray | float | object]:
    """Fit (mu, alpha) from observed counts using Poisson-approx likelihood.

    Uses :func:`motac.sim.hawkes_loglik_poisson_observed` which approximates the
    simulator observation model by

        y_obs(t) ~ Poisson(p_detect * lambda(t) + false_rate).

    Notes
    -----
    The history term for lambda(t) is computed from `y_true_for_history`.
    In real-data settings you may substitute y_obs, but then this is a
    misspecified likelihood.
    """

    if y_obs.shape != y_true_for_history.shape:
        raise ValueError("y_obs and y_true_for_history must have same shape")

    if y_true_for_history.ndim != 2:
        raise ValueError("y_true_for_history must be 2D")
    n_locations, _ = y_true_for_history.shape
    if n_locations != world.n_locations:
        raise ValueError("y_true_for_history first dimension must match world.n_locations")

    y_mean = y_obs.mean(axis=1)
    mu0 = (y_mean if init_mu is None else np.asarray(init_mu, dtype=float)).copy()
    if mu0.shape != (n_locations,):
        raise ValueError("init_mu must have shape (n_locations,)")
    mu0 = np.clip(mu0, 1e-6, None)
    alpha0 = float(max(init_alpha, 0.0))

    theta0 = np.concatenate(
        [
            np.log(np.expm1(mu0) + 1e-6),
            np.array([np.log(np.expm1(alpha0) + 1e-6)]),
        ]
    )

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta_mu = theta[:n_locations]
        theta_alpha = theta[n_locations]
        mu = _softplus(theta_mu) + 1e-12
        alpha = float(_softplus(np.array([theta_alpha]))[0])
        return mu, alpha

    mu_init, alpha_init = unpack(theta0)
    ll_init = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=mu_init,
        alpha=alpha_init,
        y_true_for_history=y_true_for_history,
        y_obs=y_obs,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    def objective(theta: np.ndarray) -> float:
        mu, alpha = unpack(theta)
        return -hawkes_loglik_poisson_observed(
            world=world,
            kernel=kernel,
            mu=mu,
            alpha=alpha,
            y_true_for_history=y_true_for_history,
            y_obs=y_obs,
            p_detect=p_detect,
            false_rate=false_rate,
        )

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    theta_hat = np.asarray(res.x, dtype=float)
    mu_hat, alpha_hat = unpack(theta_hat)
    ll = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=mu_hat,
        alpha=alpha_hat,
        y_true_for_history=y_true_for_history,
        y_obs=y_obs,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    return {
        "mu": mu_hat,
        "alpha": float(alpha_hat),
        "loglik": float(ll),
        "loglik_init": float(ll_init),
        "result": res,
    }
