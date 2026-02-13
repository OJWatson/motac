from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .likelihood import road_loglik
from .neural_kernels import KernelFn


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def fit_road_hawkes_mle(
    *,
    travel_time_s: sp.csr_matrix,
    kernel: np.ndarray,
    y: np.ndarray,
    family: str = "poisson",
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    init_beta: float = 1e-3,
    init_dispersion: float = 10.0,
    kernel_fn: KernelFn | None = None,
    validate_kernel: bool = True,
    maxiter: int = 600,
) -> dict[str, object]:
    """Fit (mu, alpha, beta) (and optionally dispersion) for road-constrained model.

    This is an M3 MVP fitter using MLE and a sparse W(d_travel) kernel.

    Parameters
    ----------
    travel_time_s:
        CSR travel-time neighbourhood matrix.
    kernel:
        Discrete lag kernel.
    y:
        Count matrix (n_cells, n_steps).
    family:
        "poisson" or "negbin".
    kernel_fn:
        Optional travel-time kernel function W(d_travel) overriding exp(-beta*d).
        If provided, it is validated via `validate_kernel_fn` by default.

    Returns
    -------
    dict with fitted parameters and optimisation result.
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D")
    n_cells, _ = y.shape

    mu0 = (
        np.asarray(y.mean(axis=1), dtype=float)
        if init_mu is None
        else np.asarray(init_mu, dtype=float)
    )
    if mu0.shape != (n_cells,):
        raise ValueError("init_mu must have shape (n_cells,)")

    mu0 = np.clip(mu0, 1e-6, None)
    alpha0 = float(max(init_alpha, 0.0))
    beta0 = float(max(init_beta, 1e-12))

    # Unconstrained: mu, alpha, beta, (dispersion)
    theta_mu0 = np.log(np.expm1(mu0) + 1e-6)
    theta_alpha0 = np.log(np.expm1(alpha0) + 1e-6)
    theta_beta0 = np.log(np.expm1(beta0) + 1e-6)

    if family == "negbin":
        disp0 = float(max(init_dispersion, 1e-6))
        theta_disp0 = np.log(np.expm1(disp0) + 1e-6)
        theta0 = np.concatenate([theta_mu0, [theta_alpha0, theta_beta0, theta_disp0]])
    else:
        theta0 = np.concatenate([theta_mu0, [theta_alpha0, theta_beta0]])

    def unpack(theta: np.ndarray):
        mu = _softplus(theta[:n_cells]) + 1e-12
        alpha = float(_softplus(theta[n_cells : n_cells + 1])[0])
        beta = float(_softplus(theta[n_cells + 1 : n_cells + 2])[0] + 1e-12)
        if family == "negbin":
            disp = float(_softplus(theta[n_cells + 2 : n_cells + 3])[0] + 1e-12)
            return mu, alpha, beta, disp
        return mu, alpha, beta, None

    mu_init, alpha_init, beta_init, disp_init = unpack(theta0)
    ll_init = road_loglik(
        travel_time_s=travel_time_s,
        mu=mu_init,
        alpha=alpha_init,
        beta=beta_init,
        kernel=kernel,
        y=y,
        family=family,
        dispersion=disp_init,
        kernel_fn=kernel_fn,
        validate_kernel=validate_kernel,
    )

    def objective(theta: np.ndarray) -> float:
        mu, alpha, beta, disp = unpack(theta)
        return -road_loglik(
            travel_time_s=travel_time_s,
            mu=mu,
            alpha=alpha,
            beta=beta,
            kernel=kernel,
            y=y,
            family=family,
            dispersion=disp,
            kernel_fn=kernel_fn,
            validate_kernel=validate_kernel,
        )

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )

    mu_hat, alpha_hat, beta_hat, disp_hat = unpack(np.asarray(res.x, dtype=float))
    ll = road_loglik(
        travel_time_s=travel_time_s,
        mu=mu_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        kernel=kernel,
        y=y,
        family=family,
        dispersion=disp_hat,
        kernel_fn=kernel_fn,
        validate_kernel=validate_kernel,
    )

    out: dict[str, object] = {
        "mu": mu_hat,
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "loglik": float(ll),
        "loglik_init": float(ll_init),
        "result": res,
        "family": family,
    }
    if family == "negbin":
        out["dispersion"] = float(disp_hat)

    return out
