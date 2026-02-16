from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .fit import fit_road_hawkes_mle
from .simulate import simulate_road_hawkes_counts


@dataclass(frozen=True)
class ParameterRecoverySummary:
    """Summary statistics for a multi-seed parameter recovery run."""

    seeds: tuple[int, ...]
    mu_true: np.ndarray
    alpha_true: float
    beta_true: float
    mu_hat: np.ndarray  # (n_seeds, n_cells)
    alpha_hat: np.ndarray  # (n_seeds,)
    beta_hat: np.ndarray  # (n_seeds,)
    loglik: np.ndarray  # (n_seeds,)
    loglik_init: np.ndarray  # (n_seeds,)

    @property
    def n_seeds(self) -> int:
        return int(self.mu_hat.shape[0])

    def mu_mae_per_seed(self) -> np.ndarray:
        return np.mean(np.abs(self.mu_hat - self.mu_true[None, :]), axis=1)

    def alpha_abs_err(self) -> np.ndarray:
        return np.abs(self.alpha_hat - float(self.alpha_true))

    def beta_abs_err(self) -> np.ndarray:
        return np.abs(self.beta_hat - float(self.beta_true))


def run_parameter_recovery_road_hawkes_poisson(
    *,
    travel_time_s: sp.csr_matrix,
    kernel: np.ndarray,
    mu_true: np.ndarray,
    alpha_true: float,
    beta_true: float,
    n_steps: int,
    seeds: list[int] | tuple[int, ...],
    maxiter: int = 600,
) -> ParameterRecoverySummary:
    """Run a small multi-seed parameter recovery harness for the M3 road Hawkes fitter.

    This utility is intentionally CI-safe and lightweight:
    - tiny synthetic substrate
    - Poisson family
    - multi-seed to reduce flakiness

    Returns a summary object with per-seed fitted parameters and simple error metrics.
    """

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    mu_true = np.asarray(mu_true, dtype=float)
    if mu_true.ndim != 1:
        raise ValueError("mu_true must be 1D")

    seeds_t = tuple(int(s) for s in seeds)
    if len(seeds_t) == 0:
        raise ValueError("seeds must be non-empty")

    n_cells = int(mu_true.shape[0])
    mu_hat = np.zeros((len(seeds_t), n_cells), dtype=float)
    alpha_hat = np.zeros((len(seeds_t),), dtype=float)
    beta_hat = np.zeros((len(seeds_t),), dtype=float)
    loglik = np.zeros((len(seeds_t),), dtype=float)
    loglik_init = np.zeros((len(seeds_t),), dtype=float)

    for i, seed in enumerate(seeds_t):
        y = simulate_road_hawkes_counts(
            travel_time_s=travel_time_s,
            mu=mu_true,
            alpha=float(alpha_true),
            beta=float(beta_true),
            kernel=kernel,
            T=int(n_steps),
            seed=int(seed),
            family="poisson",
        )

        fit = fit_road_hawkes_mle(
            travel_time_s=travel_time_s,
            kernel=kernel,
            y=y,
            family="poisson",
            maxiter=int(maxiter),
        )

        mu_hat[i, :] = np.asarray(fit["mu"], dtype=float)
        alpha_hat[i] = float(fit["alpha"])
        beta_hat[i] = float(fit["beta"])
        loglik[i] = float(fit["loglik"])
        loglik_init[i] = float(fit["loglik_init"])

    return ParameterRecoverySummary(
        seeds=seeds_t,
        mu_true=mu_true,
        alpha_true=float(alpha_true),
        beta_true=float(beta_true),
        mu_hat=mu_hat,
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        loglik=loglik,
        loglik_init=loglik_init,
    )
