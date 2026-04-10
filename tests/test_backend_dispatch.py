import jax.numpy as jnp
import pytest

from motac.connectivity import make_grid_edgelist, make_grid_stencil
from motac.data import GridSpec
from motac.infer import FitConfig
from motac.model import HawkesModelSpec, MobilityHawkesModel
from motac.simulate import HawkesSimParams, simulate_counts


def _toy_data(connectivity, grid, seed=0):
    marks = ("m1", "m2")
    params = HawkesSimParams(
        mu=0.02,
        alpha_from=jnp.array([0.2, 0.15], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )
    return simulate_counts(
        T=10,
        grid=grid,
        num_nodes=grid.shape[0] * grid.shape[1],
        marks=marks,
        connectivity=connectivity,
        params=params,
        seed=seed,
    )


def test_backend_edges_requires_edgelist():
    grid = GridSpec(shape=(4, 4))
    data = _toy_data(make_grid_stencil(size=3), grid, seed=1)
    model = MobilityHawkesModel(HawkesModelSpec(marks=("m1", "m2"), backend="edges", observation="poisson"))

    with pytest.raises(TypeError):
        model.fit(
            data,
            method="map_ensemble",
            config=FitConfig(map_steps=2, map_ensemble_size=1, map_patience=1),
            seed=2,
        )


def test_backend_bcoo_deferred():
    grid = GridSpec(shape=(4, 4))
    data = _toy_data(make_grid_stencil(size=3), grid, seed=3)
    model = MobilityHawkesModel(HawkesModelSpec(marks=("m1", "m2"), backend="bcoo", observation="poisson"))

    with pytest.raises(NotImplementedError):
        model.fit(
            data,
            method="map_ensemble",
            config=FitConfig(map_steps=2, map_ensemble_size=1, map_patience=1),
            seed=4,
        )


def test_backend_edges_success_path():
    grid = GridSpec(shape=(4, 4))
    edges = make_grid_edgelist(grid)
    data = _toy_data(edges, grid, seed=5)
    model = MobilityHawkesModel(HawkesModelSpec(marks=("m1", "m2"), backend="edges", observation="poisson"))

    fit = model.fit(
        data,
        method="map_ensemble",
        config=FitConfig(map_steps=4, map_ensemble_size=1, map_patience=2),
        seed=6,
    )
    assert fit.method == "map_ensemble"
