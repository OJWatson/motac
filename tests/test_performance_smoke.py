import os

import jax.numpy as jnp
import pytest

from motac.connectivity import make_grid_stencil
from motac.data import GridSpec
from motac.infer import FitConfig
from motac.model import HawkesModelSpec, MobilityHawkesModel
from motac.simulate import HawkesSimParams, simulate_counts


@pytest.mark.slow
def test_100x100_short_smoke_map_forecast():
    if os.environ.get("RUN_SLOW_SMOKE", "0") != "1":
        pytest.skip("Set RUN_SLOW_SMOKE=1 to run 100x100 smoke test")

    grid = GridSpec(shape=(100, 100))
    marks = ("m1", "m2")
    params = HawkesSimParams(
        mu=0.01,
        alpha_from=jnp.array([0.12, 0.1], dtype=jnp.float32),
        P_from_to=jnp.array([[0.85, 0.15], [0.2, 0.8]], dtype=jnp.float32),
        rho=jnp.array([[0.5, 0.8], [0.45, 0.75]], dtype=jnp.float32),
        w_time=jnp.array([[0.75, 0.25], [0.65, 0.35]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )

    data = simulate_counts(
        T=8,
        grid=grid,
        num_nodes=10_000,
        marks=marks,
        connectivity=make_grid_stencil(size=3),
        params=params,
        seed=1,
    )

    model = MobilityHawkesModel(HawkesModelSpec(marks=marks, temporal_bases=2, observation="poisson"))
    fit = model.fit(
        data,
        method="map_ensemble",
        config=FitConfig(map_steps=8, map_ensemble_size=1, map_patience=4),
        seed=2,
    )
    fc = model.forecast(data, fit, horizon=7, num_samples=4, seed=3)

    assert fc.y_samples.shape == (4, 7, 10_000, 2)
