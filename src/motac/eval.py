from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_mle_alpha_mu_beta,
    hawkes_loglik_poisson,
    sample_hawkes_predictive_paths,
    simulate_hawkes_counts,
)
from .sim.world import generate_random_world


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Configuration for a small synthetic evaluation run."""

    seed: int = 0
    n_locations: int = 5
    n_steps_train: int = 60
    horizon: int = 7

    # True (simulator) parameters.
    mu: float = 0.1
    alpha: float = 0.6
    n_lags: int = 6
    beta: float = 1.0
    p_detect: float = 1.0
    false_rate: float = 0.0

    # Fit settings.
    fit_maxiter: int = 400

    # Forecast settings.
    n_paths: int = 200
    q: tuple[float, ...] = (0.05, 0.5, 0.95)

    def to_json(self) -> str:
        d = asdict(self)
        d["q"] = list(self.q)
        return json.dumps(d)

    @staticmethod
    def from_json(text: str) -> EvalConfig:
        payload = json.loads(text)
        payload["q"] = tuple(payload.get("q", [0.05, 0.5, 0.95]))
        return EvalConfig(**payload)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def evaluate_synthetic(config: EvalConfig) -> dict[str, object]:
    """Run a tiny, deterministic end-to-end backtest on synthetic data.

    Steps
    -----
    1) Generate a random world and simulate counts for train+horizon steps.
    2) Fit (mu, alpha, beta) by MLE on the training window.
    3) Produce horizon-step predictive samples and predictive summaries.
    4) Compute a minimal metric set (held-out NLL, RMSE/MAE, quantile coverage).

    Returns
    -------
    dict with keys:
      - config: dict
      - fit: dict
      - forecasts: dict
      - metrics: dict
    """

    if config.n_steps_train <= 1:
        raise ValueError("n_steps_train must be > 1")
    if config.horizon <= 0:
        raise ValueError("horizon must be positive")

    world = generate_random_world(
        n_locations=config.n_locations, seed=config.seed, lengthscale=0.5
    )
    kernel_true = discrete_exponential_kernel(n_lags=config.n_lags, beta=config.beta)

    params_true = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), float(config.mu)),
        alpha=float(config.alpha),
        kernel=kernel_true,
        p_detect=float(config.p_detect),
        false_rate=float(config.false_rate),
    )

    out = simulate_hawkes_counts(
        world=world,
        params=params_true,
        n_steps=int(config.n_steps_train + config.horizon),
        seed=int(config.seed + 1),
    )

    y_true = out["y_true"].astype(int)
    y_train = y_true[:, : config.n_steps_train]
    y_test = y_true[:, config.n_steps_train : config.n_steps_train + config.horizon]

    fit = fit_hawkes_mle_alpha_mu_beta(
        world=world,
        n_lags=int(config.n_lags),
        y=y_train,
        init_alpha=0.1,
        init_beta=1.0,
        maxiter=int(config.fit_maxiter),
    )

    params_hat = HawkesDiscreteParams(
        mu=np.asarray(fit["mu"], dtype=float),
        alpha=float(fit["alpha"]),
        kernel=np.asarray(fit["kernel"], dtype=float),
        p_detect=float(config.p_detect),
        false_rate=float(config.false_rate),
    )

    paths = sample_hawkes_predictive_paths(
        world=world,
        params=params_hat,
        y_history=y_train,
        horizon=int(config.horizon),
        n_paths=int(config.n_paths),
        seed=int(config.seed + 2),
    )

    y_paths = np.asarray(paths["y_true"], dtype=float)  # (n_paths, n_locations, horizon)
    y_mean = y_paths.mean(axis=0)
    y_quantiles = np.quantile(y_paths, np.asarray(config.q, dtype=float), axis=0)

    # Held-out NLL (conditional on sampled intensities using the *true* heldout y).
    # We evaluate the fitted latent likelihood on the full series, then subtract train part.
    ll_train = hawkes_loglik_poisson(
        world=world,
        kernel=params_hat.kernel,
        mu=params_hat.mu,
        alpha=params_hat.alpha,
        y=y_train,
    )
    ll_full = hawkes_loglik_poisson(
        world=world,
        kernel=params_hat.kernel,
        mu=params_hat.mu,
        alpha=params_hat.alpha,
        y=y_true[:, : config.n_steps_train + config.horizon],
    )
    ll_test = float(ll_full - ll_train)
    n_test = int(world.n_locations * config.horizon)
    nll_test = float(-ll_test / max(n_test, 1))

    rmse = _rmse(y_test.astype(float), y_mean)
    mae = _mae(y_test.astype(float), y_mean)

    # Calibration-ish: coverage of central interval [q_lo, q_hi] if provided.
    coverage = None
    if len(config.q) >= 2:
        q_lo = float(min(config.q))
        q_hi = float(max(config.q))
        lo = np.quantile(y_paths, q_lo, axis=0)
        hi = np.quantile(y_paths, q_hi, axis=0)
        coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))

    forecasts = {
        "y_true_mean": y_mean,
        "y_true_quantiles": y_quantiles,
        "q": np.asarray(config.q, dtype=float),
    }

    metrics: dict[str, float] = {
        "nll_test": float(nll_test),
        "rmse": float(rmse),
        "mae": float(mae),
    }
    if coverage is not None:
        metrics["coverage"] = float(coverage)

    return {
        "config": asdict(config) | {"q": list(config.q)},
        "fit": {
            "mu": np.asarray(fit["mu"], dtype=float),
            "alpha": float(fit["alpha"]),
            "beta": float(fit["beta"]),
            "loglik": float(fit["loglik"]),
            "loglik_init": float(fit["loglik_init"]),
        },
        "forecasts": forecasts,
        "metrics": metrics,
    }
