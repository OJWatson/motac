"""Posterior predictive forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .connectivity import apply_spatial_backend
from .data import CountsTensor
from .infer import FitResult
from .model import HawkesModelSpec
from .utils import sample_observation_counts


@dataclass(frozen=True)
class ForecastResult:
    """Posterior predictive forecast container.

    Attributes
    ----------
    y_samples:
        Predictive draws with shape `[S, H, J, M]`.
    mean:
        Monte Carlo mean of `y_samples` over sample axis.
    quantiles:
        Quantile summaries keyed by quantile level.
    meta:
        Metadata such as horizon and sample count.
    """

    y_samples: jnp.ndarray
    mean: jnp.ndarray
    quantiles: dict[float, jnp.ndarray]
    meta: dict[str, Any]


def forecast_from_fit(
    *,
    data: CountsTensor,
    spec: HawkesModelSpec,
    fit: FitResult,
    horizon: int,
    num_samples: int,
    seed: int,
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> ForecastResult:
    """Generate posterior predictive trajectories from a fitted model.

    The rollout replays training history to recover terminal latent state and
    then performs autoregressive simulation for `horizon` steps.
    """

    y_train = data.y.astype(jnp.float32)
    y_last = y_train[-1]
    _, j_nodes, m = y_train.shape
    b = spec.temporal_bases
    samples = fit.posterior_samples

    num_post = samples["mu"].shape[0]

    # Recover final excitation state by replaying training history.
    def update_h(carry: jnp.ndarray, y_t: jnp.ndarray, rho: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_state = rho[None, :, :] * carry + y_t[:, :, None]
        return h_state, h_state

    def one_draw(draw_key: jax.Array) -> jnp.ndarray:
        idx = jax.random.randint(draw_key, (), 0, num_post)
        mu = samples["mu"][idx]
        alpha = samples["alpha"][idx]
        p = samples["p"][idx]
        rho = samples["rho"][idx]
        w = samples["w"][idx]
        phi = samples["phi"][idx]

        g = alpha[:, None] * p
        mu_jm = jnp.broadcast_to(mu[None, :], (j_nodes, m))
        phi_jm = jnp.broadcast_to(phi[None, :], (j_nodes, m))
        temporal_scale = (
            (1.0 - rho)
            if spec.normalise_time_kernel
            else jnp.ones_like(rho)
        )

        h0 = jnp.zeros((j_nodes, m, b), dtype=jnp.float32)
        history = y_train[:-1]
        h_final, _ = jax.lax.scan(
            lambda c, yt: update_h(c, yt, rho), h0, history)

        def step(carry: tuple[jnp.ndarray, jnp.ndarray, jax.Array], _h: jnp.ndarray):
            h_state, y_prev, key_state = carry
            h_state = rho[None, :, :] * h_state + y_prev[:, :, None]
            h_flat = h_state.reshape(j_nodes, m * b)
            h_spatial = apply_spatial_backend(
                h_flat,
                connectivity=data.connectivity,
                grid=data.grid,
                backend=spec.backend,
                normalise_stencil=spec.normalise_spatial_kernel,
            ).reshape(j_nodes, m, b)
            mixed = jnp.sum(
                w[None, :, :] * temporal_scale[None, :, :] * h_spatial,
                axis=-1,
            )
            lam = jnp.clip(mu_jm + mixed @ g, 1e-8, 1e12)
            key_state, sub = jax.random.split(key_state)
            y_t = sample_observation_counts(
                sub,
                lam,
                observation=spec.observation,
                concentration=phi_jm,
            ).astype(jnp.float32)
            return (h_state, y_t, key_state), y_t

        init = (h_final, y_last, jax.random.fold_in(draw_key, 999))
        (_, _, _), ys = jax.lax.scan(step, init, jnp.arange(horizon))
        return ys

    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)
    y_samples = jax.vmap(one_draw)(keys)
    mean = jnp.mean(y_samples, axis=0)
    quantiles = {
        float(q): jnp.quantile(y_samples, float(q), axis=0)
        for q in quantile_levels
    }

    return ForecastResult(
        y_samples=y_samples,
        mean=mean,
        quantiles=quantiles,
        meta={
            "horizon": horizon,
            "num_samples": num_samples,
            "quantile_levels": tuple(float(q) for q in quantile_levels),
        },
    )
