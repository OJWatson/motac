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
    """One-step-ahead intensity forecast given count history.

    This matches the simulator recursion exactly: the returned intensities are
    the conditional Poisson means for the *next* time step.

    Examples
    --------
    >>> import numpy as np
    >>> from motac.sim import generate_random_world
    >>> from motac.sim.hawkes import HawkesDiscreteParams, discrete_exponential_kernel
    >>> world = generate_random_world(n_locations=3, seed=0, lengthscale=0.5)
    >>> params = HawkesDiscreteParams(
    ...     mu=np.full((3,), 0.1),
    ...     alpha=0.5,
    ...     kernel=discrete_exponential_kernel(n_lags=4, beta=1.0),
    ... )
    >>> y_hist = np.zeros((3, 10), dtype=int)
    >>> lam_next = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y_hist)
    >>> lam_next.shape
    (3,)

    Parameters
    ----------
    world:
        Provides the mobility matrix.
    params:
        Hawkes parameters.
    y_history:
        Past counts with shape ``(n_locations, n_steps_history)``.
        The forecast is for time ``t = n_steps_history``.

    Returns
    -------
    lam_next:
        Non-negative intensity (Poisson mean) for the next step with shape
        ``(n_locations,)``.
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

    The future counts are unknown. This function produces a deterministic
    forecast by iterating the Hawkes recursion and substituting the conditional
    expectations for future counts, i.e. it rolls forward with ``y(t) := lambda(t)``.

    This is cheap and stable enough for quick QA, but it is *not* a sample from
    the full predictive distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from motac.sim import generate_random_world
    >>> from motac.sim.hawkes import HawkesDiscreteParams, discrete_exponential_kernel
    >>> world = generate_random_world(n_locations=2, seed=0, lengthscale=0.5)
    >>> params = HawkesDiscreteParams(
    ...     mu=np.full((2,), 0.2),
    ...     alpha=0.3,
    ...     kernel=discrete_exponential_kernel(n_lags=3, beta=1.0),
    ... )
    >>> y_hist = np.zeros((2, 5), dtype=int)
    >>> lam = predict_hawkes_intensity_multi_step(
    ...     world=world, params=params, y_history=y_hist, horizon=4
    ... )
    >>> lam.shape
    (2, 4)

    Parameters
    ----------
    y_history:
        Past counts with shape ``(n_locations, n_steps_history)``.
    horizon:
        Number of steps to forecast ahead.

    Returns
    -------
    intensity:
        Forecast intensities with shape ``(n_locations, horizon)`` for times
        ``t = n_steps_history .. n_steps_history + horizon - 1``.
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


def sample_hawkes_predictive_paths(
    *,
    world: World,
    params: HawkesDiscreteParams,
    y_history: np.ndarray,
    horizon: int,
    n_paths: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample predictive paths forward from a history.

    This provides an uncertainty hook beyond the deterministic
    :func:`predict_hawkes_intensity_multi_step` expected-count roll-forward.

    It simulates the latent counts y_true forward using the Hawkes recursion,
    and then applies the same observation model used in
    :func:`simulate_hawkes_counts` to produce y_obs.

    Parameters
    ----------
    y_history:
        Past *latent* counts of shape (n_locations, n_steps_history) used to
        seed the Hawkes history. (For real data you may substitute y_obs, but
        that is misspecified.)
    horizon:
        Forecast horizon (number of steps ahead).
    n_paths:
        Number of Monte Carlo paths.

    Returns
    -------
    dict with arrays:
      - y_true: (n_paths, n_locations, horizon)
      - y_obs:  (n_paths, n_locations, horizon)
      - intensity: (n_paths, n_locations, horizon)
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if y_history.ndim != 2:
        raise ValueError("y_history must be 2D (n_locations, n_steps)")
    if y_history.shape[0] != world.n_locations:
        raise ValueError("y_history first dimension must match world.n_locations")

    rng = np.random.default_rng(seed)

    n_locations = world.n_locations
    y_hist = np.asarray(y_history, dtype=int)

    y_true_paths = np.zeros((n_paths, n_locations, horizon), dtype=int)
    y_obs_paths = np.zeros((n_paths, n_locations, horizon), dtype=int)
    intensity_paths = np.zeros((n_paths, n_locations, horizon), dtype=float)

    for p in range(n_paths):
        # Extend history step by step for this path.
        y_ext = y_hist.copy()
        for k in range(horizon):
            t = int(y_ext.shape[1])
            h = _convolved_history(y_ext, params.kernel, t)
            excitation = world.mobility @ h
            lam = params.mu + params.alpha * excitation
            lam = np.clip(lam, 0.0, None)

            intensity_paths[p, :, k] = lam
            y_next = rng.poisson(lam=lam)
            y_true_paths[p, :, k] = y_next

            # Observation model.
            y_det = (
                rng.binomial(n=y_next, p=params.p_detect)
                if params.p_detect < 1.0
                else y_next
            )

            if params.false_rate > 0.0:
                y_fp = rng.poisson(lam=params.false_rate, size=y_next.shape)
            else:
                y_fp = np.zeros_like(y_next)

            y_obs_paths[p, :, k] = y_det + y_fp

            # Append latent to history.
            y_ext = np.concatenate([y_ext, y_next.reshape(n_locations, 1)], axis=1)

    return {
        "y_true": y_true_paths,
        "y_obs": y_obs_paths,
        "intensity": intensity_paths,
    }


def sample_hawkes_observed_predictive_paths_poisson_approx(
    *,
    world: World,
    mu: np.ndarray,
    alpha: float,
    kernel: np.ndarray,
    y_history_for_intensity: np.ndarray,
    horizon: int,
    n_paths: int,
    seed: int,
    p_detect: float,
    false_rate: float,
) -> dict[str, np.ndarray]:
    """Sample predictive *observed* count paths with Poisson approximation.

    This is aligned with the Poisson-approx likelihood used in
    :func:`motac.sim.hawkes_loglik_poisson_observed`:

        y_obs(t) ~ Poisson(p_detect * lambda(t) + false_rate)

    where lambda(t) is the latent Hawkes intensity.

    The latent counts are *not* sampled; instead we sample y_obs directly given
    the intensity recursion driven by `y_history_for_intensity`.

    Returns
    -------
    dict with arrays:
      - y_obs: (n_paths, n_locations, horizon)
      - intensity_obs: (n_paths, n_locations, horizon)
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if not (0.0 < p_detect <= 1.0):
        raise ValueError("p_detect must be in (0,1]")
    if false_rate < 0.0:
        raise ValueError("false_rate must be non-negative")

    if y_history_for_intensity.ndim != 2:
        raise ValueError("y_history_for_intensity must be 2D")
    if y_history_for_intensity.shape[0] != world.n_locations:
        raise ValueError("history first dimension must match world.n_locations")

    rng = np.random.default_rng(seed)
    n_locations = world.n_locations

    y_obs_paths = np.zeros((n_paths, n_locations, horizon), dtype=int)
    intensity_obs_paths = np.zeros((n_paths, n_locations, horizon), dtype=float)

    y_hist = np.asarray(y_history_for_intensity, dtype=float)

    for p in range(n_paths):
        y_ext = y_hist.copy()
        for k in range(horizon):
            t = int(y_ext.shape[1])
            h = _convolved_history(y_ext, kernel, t)
            excitation = world.mobility @ h
            lam_true = mu + alpha * excitation
            lam_true = np.clip(lam_true, 0.0, None)

            lam_obs = p_detect * lam_true + false_rate
            lam_obs = np.clip(lam_obs, 0.0, None)

            intensity_obs_paths[p, :, k] = lam_obs
            y_next_obs = rng.poisson(lam=lam_obs)
            y_obs_paths[p, :, k] = y_next_obs

            # Roll-forward with expected observed counts for stability.
            y_ext = np.concatenate([y_ext, lam_obs.reshape(n_locations, 1)], axis=1)

    return {
        "y_obs": y_obs_paths,
        "intensity_obs": intensity_obs_paths,
    }
