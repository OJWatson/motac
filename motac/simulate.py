"""Simulation routines for discrete-time mobility Hawkes models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .connectivity import ConvStencil, EdgeList, apply_spatial_backend
from .data import CountsTensor, EventTable, GridSpec
from .utils import sample_observation_counts


@dataclass(frozen=True)
class HawkesSimParams:
    """Parameters controlling simulation dynamics and observation sampling."""

    mu: float | jnp.ndarray
    alpha_from: jnp.ndarray
    P_from_to: jnp.ndarray
    rho: jnp.ndarray
    w_time: jnp.ndarray
    obs: str = "nb2"
    concentration: float | jnp.ndarray = 20.0
    zi_gate: float | None = None


@dataclass(frozen=True)
class HawkesSimNoise:
    """Optional post-processing perturbations applied to simulated counts."""

    thinning_p: float = 1.0
    jitter_time_p: float = 0.0
    jitter_space_p: float = 0.0
    rng_streams: int = 3


def _neighbour_nodes(node: int, grid: GridSpec) -> list[int]:
    """Return 4-neighbour node indices for a flattened grid node id."""

    h, w = grid.shape
    r, c = divmod(node, w)
    neighbours: list[int] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            neighbours.append(rr * w + cc)
    return neighbours


def _apply_noise_postprocess(
    y: np.ndarray,
    *,
    noise: HawkesSimNoise,
    seed: int,
    grid: GridSpec | None,
) -> np.ndarray:
    """Apply optional post-simulation observation noise transforms."""
    out = np.asarray(y, dtype=np.int32).copy()
    rng = np.random.default_rng(seed + 10_000)

    if noise.thinning_p < 1.0:
        p = float(np.clip(noise.thinning_p, 0.0, 1.0))
        out = rng.binomial(out, p).astype(np.int32)

    if noise.jitter_time_p > 0.0:
        p = float(np.clip(noise.jitter_time_p, 0.0, 1.0))
        t_len = out.shape[0]
        moved = np.zeros_like(out)
        k = rng.binomial(out, p).astype(np.int32)
        out -= k
        if t_len > 1:
            moved[1] += k[0]
            moved[t_len - 2] += k[t_len - 1]
            if t_len > 2:
                interior = k[1: t_len - 1]
                k_prev = rng.binomial(interior, 0.5).astype(np.int32)
                k_next = interior - k_prev
                moved[0: t_len - 2] += k_prev
                moved[2:t_len] += k_next
        else:
            moved[0] += k[0]
        out += moved

    if noise.jitter_space_p > 0.0 and grid is not None:
        p = float(np.clip(noise.jitter_space_p, 0.0, 1.0))
        t_len, j_len, m_len = out.shape
        moved = np.zeros_like(out)
        neigh_cache = {j: _neighbour_nodes(j, grid) for j in range(j_len)}
        k = rng.binomial(out, p).astype(np.int32)
        out -= k
        for t in range(t_len):
            for j in range(j_len):
                neigh = neigh_cache[j]
                if not neigh:
                    continue
                for m in range(m_len):
                    count_move = int(k[t, j, m])
                    if count_move <= 0:
                        continue
                    dst = rng.choice(neigh, size=count_move)
                    moved[t, :, m] += np.bincount(dst,
                                                  minlength=j_len).astype(np.int32)
        out += moved

    return out


def simulate_counts(
    *,
    T: int,
    grid: GridSpec | None,
    num_nodes: int,
    marks: Sequence[str],
    connectivity: ConvStencil | EdgeList,
    params: HawkesSimParams,
    noise: HawkesSimNoise = HawkesSimNoise(),
    seed: int = 0,
) -> CountsTensor:
    """Simulate a marked mobility-constrained Hawkes count tensor.

    The simulator runs the same latent recurrence as the model and samples from
    the requested observation family (`poisson` or `nb2`) at each step.
    """

    m = len(marks)
    b = params.rho.shape[1]

    mu = jnp.asarray(params.mu, dtype=jnp.float32)
    if mu.ndim == 0:
        mu = jnp.full((num_nodes, m), mu)
    elif mu.ndim == 1:
        mu = jnp.broadcast_to(mu[None, :], (num_nodes, m))

    alpha_from = jnp.asarray(params.alpha_from, dtype=jnp.float32)
    p_from_to = jnp.asarray(params.P_from_to, dtype=jnp.float32)
    g = alpha_from[:, None] * p_from_to

    rho = jnp.asarray(params.rho, dtype=jnp.float32)
    w_time = jnp.asarray(params.w_time, dtype=jnp.float32)

    concentration_jm: jnp.ndarray | None = None
    if params.obs == "nb2":
        concentration = jnp.asarray(params.concentration, dtype=jnp.float32)
        if concentration.ndim == 0:
            concentration = jnp.full((m,), concentration)
        concentration_jm = jnp.broadcast_to(
            concentration[None, :], (num_nodes, m))

    key = jax.random.PRNGKey(seed)

    def step(carry: tuple[jnp.ndarray, jax.Array, jnp.ndarray], _t: jnp.ndarray):
        h_state, key_state, y_prev = carry

        h_state = rho[None, :, :] * h_state + y_prev[:, :, None]
        h_flat = h_state.reshape(num_nodes, m * b)
        h_spatial = apply_spatial_backend(
            h_flat,
            connectivity=connectivity,
            grid=grid,
            backend="auto",
        ).reshape(num_nodes, m, b)
        base_mix = jnp.sum(w_time[None, :, :] * h_spatial, axis=-1)
        excitation = base_mix @ g
        lam = mu + excitation

        key_state, sub = jax.random.split(key_state)
        y_t = sample_observation_counts(
            sub,
            lam,
            observation=params.obs,
            concentration=concentration_jm,
        )
        y_t = jnp.clip(y_t, 0.0, 1e6).astype(jnp.float32)
        return (h_state, key_state, y_t), y_t

    init_h = jnp.zeros((num_nodes, m, b), dtype=jnp.float32)
    init_y = jnp.zeros((num_nodes, m), dtype=jnp.float32)
    (_, _, _), ys = jax.lax.scan(step, (init_h, key, init_y), xs=jnp.arange(T))
    ys_np = _apply_noise_postprocess(
        np.asarray(ys), noise=noise, seed=seed, grid=grid)

    return CountsTensor(
        y=jnp.asarray(ys_np, dtype=jnp.float32),
        t0=np.datetime64("2000-01-01"),
        dt_days=1,
        num_time=T,
        num_nodes=num_nodes,
        marks=tuple(marks),
        node_coords=None,
        grid=grid,
        connectivity=connectivity,
        covariates={},
    )


def counts_to_events(
    counts: CountsTensor,
    *,
    include_zero: bool = False,
) -> EventTable:
    """Expand count tensor into event rows for plotting/debugging.

    Each count at `(t, node, mark)` is expanded into repeated rows with unit weight.
    """
    y = np.asarray(counts.y)
    t_len, j_len, m_len = y.shape

    rows: list[dict[str, object]] = []
    for t in range(t_len):
        day = counts.t0 + np.timedelta64(t * counts.dt_days, "D")
        for j in range(j_len):
            for m in range(m_len):
                c = int(y[t, j, m])
                if c <= 0 and not include_zero:
                    continue
                reps = max(c, 1) if include_zero else c
                for _ in range(reps):
                    rows.append(
                        {
                            "time": str(day),
                            "node": j,
                            "mark": counts.marks[m],
                            "count": c,
                            "weight": 1.0 if c > 0 else 0.0,
                        }
                    )

    import pandas as pd

    return EventTable(
        df=pd.DataFrame(rows),
        time_col="time",
        lat_col="node",
        lon_col="node",
        mark_col="mark",
        weight_col="weight",
    )
