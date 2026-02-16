"""Likelihood primitives for differentiable inference.

Task M3.1: provide Poisson and Negative Binomial (NB2) likelihoods with a JAX
implementation (when available) and a NumPy/SciPy fallback.

The functions are written to be shape-preserving and work on either NumPy
arrays or JAX arrays. When JAX is present and any input is a JAX array, the JAX
path is used.

Parameterisations
-----------------
Poisson:
    Y ~ Pois(mean)

Negative binomial (NB2):
    Var[Y] = mean + mean^2 / dispersion,
    with dispersion > 0 and larger values approaching Poisson.

This matches :mod:`motac.model.likelihood` so the model and inference stacks
agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln as _gammaln_np

try:  # optional
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as _gammaln_jax

    _HAS_JAX = True
except Exception:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _gammaln_jax = None  # type: ignore[assignment]
    _HAS_JAX = False


def _use_jax(*xs: Any) -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(isinstance(x, jnp.ndarray) for x in xs if x is not None)
    except Exception:  # pragma: no cover
        return False


def poisson_logpmf(*, y: Any, mean: Any, eps: float = 1e-12) -> Any:
    """Poisson log PMF with safe log(mean)."""

    if _use_jax(y, mean):
        y = jnp.asarray(y, dtype=jnp.float32)
        m = jnp.asarray(mean, dtype=jnp.float32)
        if y.shape != m.shape:
            raise ValueError("y and mean must have the same shape")
        m_safe = jnp.clip(m, eps, None)
        return y * jnp.log(m_safe) - m_safe - _gammaln_jax(y + 1.0)

    y = np.asarray(y, dtype=float)
    m = np.asarray(mean, dtype=float)
    if y.shape != m.shape:
        raise ValueError("y and mean must have the same shape")
    m_safe = np.clip(m, eps, None)
    return y * np.log(m_safe) - m_safe - _gammaln_np(y + 1.0)


def negbin_logpmf(*, y: Any, mean: Any, dispersion: float) -> Any:
    """Negative binomial (NB2) log PMF parameterised by mean and dispersion."""

    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    if _use_jax(y, mean):
        y = jnp.asarray(y, dtype=jnp.float32)
        m = jnp.asarray(mean, dtype=jnp.float32)
        if y.shape != m.shape:
            raise ValueError("y and mean must have the same shape")

        k = jnp.asarray(dispersion, dtype=jnp.float32)
        log_coeff = _gammaln_jax(y + k) - _gammaln_jax(k) - _gammaln_jax(y + 1.0)
        log_p = k * (jnp.log(k) - jnp.log(k + m))
        log_q = y * (jnp.log(m) - jnp.log(k + m))
        return log_coeff + log_p + log_q

    y = np.asarray(y, dtype=float)
    m = np.asarray(mean, dtype=float)
    if y.shape != m.shape:
        raise ValueError("y and mean must have the same shape")

    k = float(dispersion)
    log_coeff = _gammaln_np(y + k) - _gammaln_np(k) - _gammaln_np(y + 1.0)
    log_p = k * (np.log(k) - np.log(k + m))
    log_q = y * (np.log(m) - np.log(k + m))
    return log_coeff + log_p + log_q


def poisson_loglik(*, y: Any, mean: Any, eps: float = 1e-12) -> Any:
    """Sum of Poisson log PMFs over all elements."""

    ll = poisson_logpmf(y=y, mean=mean, eps=eps)
    if _use_jax(ll):
        return jnp.sum(ll)
    return float(np.sum(ll))


def negbin_loglik(*, y: Any, mean: Any, dispersion: float) -> Any:
    """Sum of NB2 log PMFs over all elements."""

    ll = negbin_logpmf(y=y, mean=mean, dispersion=dispersion)
    if _use_jax(ll):
        return jnp.sum(ll)
    return float(np.sum(ll))
