from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

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


def hawkes_loglik_poisson_observed(
    *,
    world: World,
    kernel: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    y_true_for_history: np.ndarray,
    y_obs: np.ndarray,
    p_detect: float,
    false_rate: float,
    eps: float = 1e-12,
) -> float:
    """Approximate log-likelihood for observed counts with detection+clutter.

    Simulator observation model
    ---------------------------
    In :func:`motac.sim.simulate_hawkes_counts`, the observed counts are

        y_obs = Binomial(y_true, p_detect) + Poisson(false_rate)

    Conditional on the latent intensity lambda(t), if we ignore the discreteness
    of thinning and treat the thinned process as Poisson, we obtain the
    approximation

        y_obs(t) ~ Poisson(p_detect * lambda(t) + false_rate).

    This is a cheap likelihood used for parameter recovery/QA.

    Parameters
    ----------
    y_true_for_history:
        Counts used to construct the Hawkes history term. For fully observed
        experiments you can pass y_obs here, but in the simulator setting this
        is typically y_true.
    y_obs:
        Observed counts.
    p_detect:
        Detection probability in (0,1].
    false_rate:
        Additive Poisson clutter rate (>=0).

    Notes
    -----
    - This is not the exact likelihood under Binomial thinning.
    - For numerical safety we lower-bound the observed intensity by `eps`.
    """

    if not (0.0 < p_detect <= 1.0):
        raise ValueError("p_detect must be in (0,1]")
    if false_rate < 0.0:
        raise ValueError("false_rate must be non-negative")
    if y_obs.shape != y_true_for_history.shape:
        raise ValueError("y_obs and y_true_for_history must have the same shape")

    lam_true = hawkes_intensity(
        world=world,
        kernel=kernel,
        mu=mu,
        alpha=alpha,
        y=y_true_for_history,
    )
    lam_obs = p_detect * lam_true + false_rate
    lam_obs = np.clip(lam_obs, eps, None)

    ll = (y_obs * np.log(lam_obs) - lam_obs - gammaln(y_obs + 1.0)).sum()
    return float(ll)


def hawkes_loglik_observed_exact(
    *,
    world: World,
    kernel: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    y_true_for_history: np.ndarray,
    y_true: np.ndarray,
    y_obs: np.ndarray,
    p_detect: float,
    false_rate: float,
) -> float:
    """Exact log-likelihood for observed counts under thinning + clutter.

    Observation model (simulator)
    -----------------------------

        y_obs = y_det + y_fp
        y_det | y_true ~ Binomial(y_true, p_detect)
        y_fp ~ Poisson(false_rate)

    This function evaluates p(y_obs | y_true) exactly by summing out y_det:

        p(y_obs | y_true) = sum_{k=0..y_obs} Binom(k|y_true,p) * Pois(y_obs-k|false_rate)

    using stable log-space logsumexp.

    Notes
    -----
    - The Hawkes parameters are accepted for API consistency and to allow
      optional extensions; the exact likelihood here depends on the latent
      counts y_true (and not directly on mu/alpha/kernel), but we validate
      shapes against the provided world.
    """

    if not (0.0 < p_detect <= 1.0):
        raise ValueError("p_detect must be in (0,1]")
    if false_rate < 0.0:
        raise ValueError("false_rate must be non-negative")

    if y_obs.shape != y_true.shape:
        raise ValueError("y_obs and y_true must have the same shape")
    if y_true_for_history.shape != y_true.shape:
        raise ValueError("y_true_for_history and y_true must have the same shape")

    if y_true.ndim != 2:
        raise ValueError("y_true must be 2D (n_locations, n_steps)")
    if y_true.shape[0] != world.n_locations:
        raise ValueError("y_true first dimension must match world.n_locations")

    # Validate Hawkes arrays lightly (even though they are not used here).
    if mu.shape != (world.n_locations,):
        raise ValueError("mu must have shape (n_locations,)")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be a non-empty 1D array")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    y_true_int = np.asarray(y_true, dtype=int)
    y_obs_int = np.asarray(y_obs, dtype=int)

    if np.any(y_true_int < 0) or np.any(y_obs_int < 0):
        raise ValueError("counts must be non-negative")

    # Work elementwise; counts are small in our toy use-cases.
    ll_total = 0.0

    # Precompute logs for Poisson.
    log_false = np.log(false_rate) if false_rate > 0.0 else -np.inf

    for yt, yo in zip(y_true_int.reshape(-1), y_obs_int.reshape(-1), strict=True):
        # Support of y_det is k in [0, min(yt, yo)].
        kmax = int(min(yt, yo))
        ks = np.arange(kmax + 1)

        # log Binom(k | yt, p) = log C(yt,k) + k log p + (yt-k) log(1-p)
        # log C(yt,k) = gammaln(yt+1) - gammaln(k+1) - gammaln(yt-k+1)
        log_choose = (
            gammaln(yt + 1.0)
            - gammaln(ks + 1.0)
            - gammaln(yt - ks + 1.0)
        )

        if p_detect == 1.0:
            log_binom = np.where(ks == yt, 0.0, -np.inf)
        else:
            logp = np.log(p_detect)
            log1mp = np.log1p(-p_detect)
            log_binom = log_choose + ks * logp + (yt - ks) * log1mp

        # log Pois(yo-k | false_rate)
        m = yo - ks
        if false_rate == 0.0:
            log_pois = np.where(m == 0, 0.0, -np.inf)
        else:
            log_pois = m * log_false - false_rate - gammaln(m + 1.0)

        ll_total += float(logsumexp(log_binom + log_pois))

    return float(ll_total)
