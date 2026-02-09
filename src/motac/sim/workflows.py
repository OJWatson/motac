from __future__ import annotations

import numpy as np

from .fit import fit_hawkes_mle_alpha_mu_observed_poisson_approx
from .hawkes import sample_hawkes_observed_predictive_paths_poisson_approx
from .predictive import summarize_predictive_paths
from .world import World


def observed_fit_sample_summarize_poisson_approx(
    *,
    world: World,
    kernel: np.ndarray,
    y_obs: np.ndarray,
    p_detect: float,
    false_rate: float,
    horizon: int,
    n_paths: int,
    seed: int,
    q: tuple[float, ...] = (0.05, 0.5, 0.95),
    init_mu: np.ndarray | None = None,
    init_alpha: float = 0.1,
    fit_maxiter: int = 600,
) -> dict[str, object]:
    """Observed-only workflow: fit -> sample -> summarize (Poisson approximation).

    This is a convenience wrapper for the default "real-data" path where latent
    counts are unavailable. It:

      1) fits (mu, alpha) with :func:`fit_hawkes_mle_alpha_mu_observed_poisson_approx`
         using `y_obs` to drive the history term;
      2) samples observed predictive paths with
         :func:`sample_hawkes_observed_predictive_paths_poisson_approx`;
      3) summarizes Monte Carlo paths with :func:`summarize_predictive_paths`.

    Returns
    -------
    dict with keys:
      - fit: dict (mu, alpha, loglik, ...)
      - paths: dict (y_obs, intensity_obs)
      - summary: dict (mean, quantiles, q)
    """

    if y_obs.ndim != 2:
        raise ValueError("y_obs must be 2D (n_locations, n_steps)")

    fit = fit_hawkes_mle_alpha_mu_observed_poisson_approx(
        world=world,
        kernel=kernel,
        y_true_for_history=y_obs,
        y_obs=y_obs,
        p_detect=p_detect,
        false_rate=false_rate,
        init_mu=init_mu,
        init_alpha=init_alpha,
        maxiter=fit_maxiter,
    )

    mu_hat = np.asarray(fit["mu"], dtype=float)
    alpha_hat = float(fit["alpha"])

    paths = sample_hawkes_observed_predictive_paths_poisson_approx(
        world=world,
        mu=mu_hat,
        alpha=alpha_hat,
        kernel=kernel,
        y_history_for_intensity=y_obs,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    summary = summarize_predictive_paths(paths=paths["y_obs"], q=q)

    return {
        "fit": fit,
        "paths": paths,
        "summary": summary,
    }
