from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .world import World


def discrete_exponential_kernel(
    *,
    n_lags: int,
    beta: float,
    normalize: bool = True,
) -> np.ndarray:
    """Create a discrete exponential kernel over lags 1..n_lags.

    Parameters
    ----------
    n_lags:
        Maximum lag length (number of steps in memory).
    beta:
        Decay rate; larger means faster decay.
    normalize:
        If True, normalize the kernel weights to sum to 1.
    """

    if n_lags <= 0:
        raise ValueError("n_lags must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")

    lags = np.arange(1, n_lags + 1, dtype=float)
    g = np.exp(-beta * (lags - 1.0))
    if normalize:
        s = float(g.sum())
        if s > 0:
            g = g / s
    return g


@dataclass(frozen=True, slots=True)
class HawkesDiscreteParams:
    """Parameters for the discrete-time Hawkes-like count model."""

    mu: np.ndarray
    alpha: float
    kernel: np.ndarray
    p_detect: float = 1.0
    false_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.kernel.ndim != 1 or self.kernel.size == 0:
            raise ValueError("kernel must be a non-empty 1D array")
        if not (0.0 < self.p_detect <= 1.0):
            raise ValueError("p_detect must be in (0,1]")
        if self.false_rate < 0:
            raise ValueError("false_rate must be non-negative")

    @property
    def n_lags(self) -> int:
        return int(self.kernel.size)

    def to_json(self) -> str:
        payload = {
            "mu": self.mu.tolist(),
            "alpha": float(self.alpha),
            "kernel": self.kernel.tolist(),
            "p_detect": float(self.p_detect),
            "false_rate": float(self.false_rate),
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(text: str) -> HawkesDiscreteParams:
        payload = json.loads(text)
        return HawkesDiscreteParams(
            mu=np.asarray(payload["mu"], dtype=float),
            alpha=float(payload["alpha"]),
            kernel=np.asarray(payload["kernel"], dtype=float),
            p_detect=float(payload.get("p_detect", 1.0)),
            false_rate=float(payload.get("false_rate", 0.0)),
        )


def _convolved_history(y: np.ndarray, kernel: np.ndarray, t: int) -> np.ndarray:
    """Compute sum_{l=1..L} kernel[l-1] * y[:, t-l]."""

    lags = kernel.size
    start = max(0, t - lags)
    # y[:, start:t] aligns with kernel for lags t-start ... 1.
    window = y[:, start:t]
    if window.size == 0:
        return np.zeros((y.shape[0],), dtype=float)

    effective = t - start
    # kernel for lags 1..effective
    k = kernel[:effective]
    # Reverse window so axis=-1 corresponds to lag 1..effective.
    return window[:, ::-1] @ k


def predict_hawkes_intensity_one_step(
    *,
    world: World,
    params: HawkesDiscreteParams,
    y_history: np.ndarray,
) -> np.ndarray:
    """One-step-ahead intensity forecast given history.

    Parameters
    ----------
    world:
        Provides the mobility matrix.
    params:
        Hawkes parameters.
    y_history:
        Array of past counts with shape (n_locations, n_steps_history).
        The forecast is for the next step t = n_steps_history.

    Returns
    -------
    lam_next:
        Non-negative intensity (Poisson mean) for the next step with shape
        (n_locations,).
    """

    if y_history.ndim != 2:
        raise ValueError("y_history must be a 2D array (n_locations, n_steps)")
    if y_history.shape[0] != world.n_locations:
        raise ValueError("y_history first dimension must match world.n_locations")
    if params.mu.shape[0] != world.n_locations:
        raise ValueError("params.mu length must match world.n_locations")

    t = int(y_history.shape[1])
    h = _convolved_history(np.asarray(y_history, dtype=float), params.kernel, t)
    excitation = world.mobility @ h
    lam = params.mu + params.alpha * excitation
    return np.clip(lam, 0.0, None)


def predict_hawkes_intensity_multi_step(
    *,
    world: World,
    params: HawkesDiscreteParams,
    y_history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Multi-step intensity forecast using expected-count roll-forward.

    This produces a deterministic forecast by iterating the Hawkes recursion
    and substituting the (conditional) expected counts for future, unobserved
    counts. Concretely, at each forecast step we set y(t) := lambda(t).

    Parameters
    ----------
    y_history:
        Past counts with shape (n_locations, n_steps_history).
    horizon:
        Number of steps to forecast ahead.

    Returns
    -------
    intensity:
        Forecast intensities of shape (n_locations, horizon) corresponding to
        times t = n_steps_history .. n_steps_history + horizon - 1.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y_history.ndim != 2:
        raise ValueError("y_history must be a 2D array (n_locations, n_steps)")
    if y_history.shape[0] != world.n_locations:
        raise ValueError("y_history first dimension must match world.n_locations")

    # Work in float since we will append expected counts.
    y_ext = np.asarray(y_history, dtype=float)
    n_locations = y_ext.shape[0]

    out = np.zeros((n_locations, horizon), dtype=float)
    for k in range(horizon):
        lam = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y_ext)
        out[:, k] = lam
        # Append expected count as proxy for the unknown future draw.
        y_ext = np.concatenate([y_ext, lam.reshape(n_locations, 1)], axis=1)

    return out


def simulate_hawkes_counts(
    *,
    world: World,
    params: HawkesDiscreteParams,
    n_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate latent and observed counts.

    Model
    -----
    For each location i and discrete time t:

        lambda_i(t) = mu_i + alpha * ( mobility @ h(t) )_i
        y_i(t) ~ Poisson(lambda_i(t))

    where h_j(t) = sum_{l=1..L} kernel[l-1] * y_j(t-l).

    Observation model:
        y_obs = Binomial(y_true, p_detect) + Poisson(false_rate)

    Returns a dict with arrays:
      - y_true: (n_locations, n_steps)
      - y_obs:  (n_locations, n_steps)
      - intensity: (n_locations, n_steps)
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if params.mu.shape[0] != world.n_locations:
        raise ValueError("params.mu length must match world.n_locations")

    rng = np.random.default_rng(seed)
    n = world.n_locations
    y_true = np.zeros((n, n_steps), dtype=int)
    intensity = np.zeros((n, n_steps), dtype=float)

    for t in range(n_steps):
        h = _convolved_history(y_true, params.kernel, t)
        excitation = world.mobility @ h
        lam = params.mu + params.alpha * excitation
        lam = np.clip(lam, 0.0, None)
        intensity[:, t] = lam
        y_true[:, t] = rng.poisson(lam=lam)

    # Observation noise.
    y_det = (
        rng.binomial(n=y_true, p=params.p_detect)
        if params.p_detect < 1.0
        else y_true.copy()
    )

    y_fp = (
        rng.poisson(lam=params.false_rate, size=y_true.shape)
        if params.false_rate > 0.0
        else np.zeros_like(y_true)
    )

    y_obs = y_det + y_fp

    return {
        "y_true": y_true,
        "y_obs": y_obs,
        "intensity": intensity,
    }
