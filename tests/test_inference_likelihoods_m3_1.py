from __future__ import annotations

import math

import numpy as np
import pytest

from motac.inference.likelihoods import (
    negbin_loglik,
    negbin_logpmf,
    poisson_loglik,
    poisson_logpmf,
)
from motac.model.likelihood import negbin_logpmf as negbin_logpmf_np
from motac.model.likelihood import poisson_logpmf as poisson_logpmf_np


def test_inference_poisson_matches_model_numpy() -> None:
    y = np.asarray([0, 1, 4], dtype=float)
    mean = np.asarray([0.25, 1.5, 3.0], dtype=float)

    ll_a = poisson_logpmf(y=y, mean=mean)
    ll_b = poisson_logpmf_np(y=y, mean=mean)

    np.testing.assert_allclose(ll_a, ll_b, rtol=0, atol=1e-12)
    assert math.isfinite(float(poisson_loglik(y=y, mean=mean)))


def test_inference_negbin_matches_model_numpy() -> None:
    y = np.asarray([0, 2, 5], dtype=float)
    mean = np.asarray([0.5, 1.25, 3.0], dtype=float)
    disp = 7.5

    ll_a = negbin_logpmf(y=y, mean=mean, dispersion=disp)
    ll_b = negbin_logpmf_np(y=y, mean=mean, dispersion=disp)

    np.testing.assert_allclose(ll_a, ll_b, rtol=0, atol=1e-12)
    assert math.isfinite(float(negbin_loglik(y=y, mean=mean, dispersion=disp)))


def _finite_diff_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x1 = x.copy()
        x1[i] += eps
        x2 = x.copy()
        x2[i] -= eps
        g[i] = (float(f(x1)) - float(f(x2))) / (2.0 * eps)
    return g


@pytest.mark.skipif(
    pytest.importorskip("importlib").util.find_spec("jax") is None,
    reason="JAX not installed",
)
def test_poisson_grad_matches_finite_diff_tiny_fixture() -> None:
    import jax
    import jax.numpy as jnp

    y = jnp.asarray([0.0, 1.0, 3.0])

    def f(theta: jnp.ndarray) -> jnp.ndarray:
        mean = jax.nn.softplus(theta)
        return poisson_loglik(y=y, mean=mean)

    theta0 = np.asarray([-0.3, 0.2, 1.0], dtype=float)
    g_ad = np.asarray(jax.grad(f)(jnp.asarray(theta0)))
    g_fd = _finite_diff_grad(lambda th: f(jnp.asarray(th)), theta0)

    np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    pytest.importorskip("importlib").util.find_spec("jax") is None,
    reason="JAX not installed",
)
def test_negbin_grad_matches_finite_diff_tiny_fixture() -> None:
    import jax
    import jax.numpy as jnp

    y = jnp.asarray([0.0, 2.0, 4.0])
    disp = 3.0

    def f(theta: jnp.ndarray) -> jnp.ndarray:
        mean = jax.nn.softplus(theta)
        return negbin_loglik(y=y, mean=mean, dispersion=disp)

    theta0 = np.asarray([-0.2, 0.5, 1.1], dtype=float)
    g_ad = np.asarray(jax.grad(f)(jnp.asarray(theta0)))
    g_fd = _finite_diff_grad(lambda th: f(jnp.asarray(th)), theta0)

    np.testing.assert_allclose(g_ad, g_fd, rtol=2e-4, atol=2e-4)
