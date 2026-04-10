"""Benchmark utilities for motac backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from .connectivity import make_grid_edgelist, make_grid_stencil
from .data import GridSpec
from .infer import FitConfig
from .model import HawkesModelSpec, MobilityHawkesModel
from .simulate import HawkesSimParams, simulate_counts


@dataclass(frozen=True)
class BenchmarkResult:
    """Runtime summary for one backend benchmark run."""

    backend: str
    simulation_seconds: float
    fit_seconds: float
    forecast_seconds: float


def _sim_params(marks: tuple[str, ...]) -> HawkesSimParams:
    """Return a shared simulation parameterization for backend comparisons."""

    m = len(marks)
    return HawkesSimParams(
        mu=0.02,
        alpha_from=jnp.full((m,), 0.2, dtype=jnp.float32),
        P_from_to=jnp.eye(m, dtype=jnp.float32),
        rho=jnp.tile(jnp.array([[0.6, 0.85]], dtype=jnp.float32), (m, 1)),
        w_time=jnp.tile(jnp.array([[0.7, 0.3]], dtype=jnp.float32), (m, 1)),
        obs="poisson",
        concentration=20.0,
    )


def benchmark_backend(
    *,
    grid_shape: tuple[int, int] = (40, 40),
    time_steps: int = 30,
    marks: tuple[str, ...] = ("a", "b"),
    backend: str = "conv",
    fit_steps: int = 40,
    seed: int = 0,
) -> BenchmarkResult:
    """Benchmark simulate-fit-forecast wall time for a single backend."""

    grid = GridSpec(shape=grid_shape)
    num_nodes = grid_shape[0] * grid_shape[1]
    params = _sim_params(marks)

    if backend == "conv":
        connectivity = make_grid_stencil(size=3)
    elif backend == "edges":
        connectivity = make_grid_edgelist(grid, neighbourhood="moore")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    t0 = time.perf_counter()
    data = simulate_counts(
        T=time_steps,
        grid=grid,
        num_nodes=num_nodes,
        marks=marks,
        connectivity=connectivity,
        params=params,
        seed=seed,
    )
    sim_s = time.perf_counter() - t0

    model = MobilityHawkesModel(HawkesModelSpec(
        marks=marks, temporal_bases=2, observation="poisson"))

    t1 = time.perf_counter()
    fit = model.fit(
        data,
        method="map_ensemble",
        config=FitConfig(map_steps=fit_steps,
                         map_ensemble_size=1, map_patience=10),
        seed=seed + 1,
    )
    fit_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    _ = model.forecast(data, fit, horizon=7, num_samples=16, seed=seed + 2)
    fc_s = time.perf_counter() - t2

    return BenchmarkResult(
        backend=backend,
        simulation_seconds=sim_s,
        fit_seconds=fit_s,
        forecast_seconds=fc_s,
    )


def compare_backends(
    *,
    grid_shape: tuple[int, int] = (40, 40),
    time_steps: int = 30,
    marks: tuple[str, ...] = ("a", "b"),
    fit_steps: int = 40,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Run the same benchmark configuration for conv and edge-list backends."""

    return [
        benchmark_backend(
            grid_shape=grid_shape,
            time_steps=time_steps,
            marks=marks,
            backend="conv",
            fit_steps=fit_steps,
            seed=seed,
        ),
        benchmark_backend(
            grid_shape=grid_shape,
            time_steps=time_steps,
            marks=marks,
            backend="edges",
            fit_steps=fit_steps,
            seed=seed + 100,
        ),
    ]


def benchmarks_to_frame(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark dataclass results into a tabular DataFrame."""
    return pd.DataFrame(
        [
            {
                "backend": r.backend,
                "simulation_seconds": r.simulation_seconds,
                "fit_seconds": r.fit_seconds,
                "forecast_seconds": r.forecast_seconds,
            }
            for r in results
        ]
    )


def save_benchmarks_csv(results: list[BenchmarkResult], path: str | Path) -> Path:
    """Save benchmark results to CSV and return the resolved output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    benchmarks_to_frame(results).to_csv(out, index=False)
    return out
