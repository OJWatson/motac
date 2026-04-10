"""Shared utility helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


EPS = 1e-8


@dataclass(frozen=True)
class ShapeSpec:
    """Canonical tensor shape metadata."""

    T: int
    J: int
    M: int


def safe_softplus(x: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """Numerically safe softplus transform with additive floor."""

    return jax.nn.softplus(x) + eps


def normalise_rows(weights: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """Row-normalize a 2D array so each row sums to one."""

    denom = jnp.sum(weights, axis=1, keepdims=True)
    return weights / (denom + eps)


def to_simplex(logits: jnp.ndarray) -> jnp.ndarray:
    """Map logits to simplex probabilities along the last axis."""

    return jax.nn.softmax(logits, axis=-1)


def check_odd(n: int, name: str) -> None:
    """Validate that an integer hyperparameter is positive and odd."""

    if n <= 0 or n % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {n}.")


def sample_nb2_gamma_poisson(
    key: jax.Array,
    lam: jnp.ndarray,
    concentration: jnp.ndarray,
) -> jnp.ndarray:
    """Sample NB2 via a gamma-poisson mixture.

    The concentration parameter follows NB2 variance convention:
    Var[Y] = mean + mean^2 / concentration.
    """
    lam = jnp.clip(lam, 1e-8, 1e12)
    gamma_key, pois_key = jax.random.split(key)
    rate = jax.random.gamma(gamma_key, concentration,
                            shape=lam.shape) * (lam / concentration)
    return jax.random.poisson(pois_key, rate)


def sample_observation_counts(
    key: jax.Array,
    lam: jnp.ndarray,
    *,
    observation: str,
    concentration: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample counts from supported observation models."""
    lam = jnp.clip(lam, 1e-8, 1e12)
    if observation == "poisson":
        return jax.random.poisson(key, lam)
    if observation == "nb2":
        if concentration is None:
            raise ValueError("concentration is required for NB2 sampling")
        return sample_nb2_gamma_poisson(key, lam, concentration)
    raise ValueError(f"Unsupported observation model: {observation}")
