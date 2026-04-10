import jax.numpy as jnp

from motac.connectivity import make_grid_stencil
from motac.data import GridSpec
from motac.infer import FitConfig
from motac.model import HawkesModelSpec, MobilityHawkesModel
from motac.simulate import HawkesSimParams, simulate_counts


def test_simulate_fit_forecast_smoke():
    grid = GridSpec(shape=(10, 10))
    marks = ("m1", "m2")
    stencil = make_grid_stencil(size=3)

    params = HawkesSimParams(
        mu=0.03,
        alpha_from=jnp.array([0.2, 0.2], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.3, 0.7]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )

    data = simulate_counts(
        T=30,
        grid=grid,
        num_nodes=100,
        marks=marks,
        connectivity=stencil,
        params=params,
        seed=0,
    )

    spec = HawkesModelSpec(marks=marks, temporal_bases=2,
                           observation="poisson")
    model = MobilityHawkesModel(spec)

    fit_cfg = FitConfig(map_steps=30, map_ensemble_size=1, map_patience=10)
    fit = model.fit(data, method="map_ensemble", config=fit_cfg, seed=0)
    fc = model.forecast(
        data,
        fit,
        horizon=5,
        num_samples=8,
        seed=1,
        quantile_levels=(0.1, 0.5, 0.9),
    )

    assert fit.method == "map_ensemble"
    assert fc.y_samples.shape == (8, 5, 100, 2)
    assert set(fc.quantiles.keys()) == {0.1, 0.5, 0.9}
    assert fc.meta["quantile_levels"] == (0.1, 0.5, 0.9)
