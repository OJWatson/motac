import jax.numpy as jnp

from motac.connectivity import make_grid_stencil
from motac.data import GridSpec
from motac.eval import rolling_backtest
from motac.infer import FitConfig
from motac.model import HawkesModelSpec, MobilityHawkesModel
from motac.simulate import HawkesSimParams, simulate_counts


def test_rolling_backtest_smoke():
    grid = GridSpec(shape=(6, 6))
    marks = ("a", "b")
    stencil = make_grid_stencil(size=3)

    params = HawkesSimParams(
        mu=0.02,
        alpha_from=jnp.array([0.2, 0.15], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )

    data = simulate_counts(
        T=24,
        grid=grid,
        num_nodes=36,
        marks=marks,
        connectivity=stencil,
        params=params,
        seed=3,
    )

    model = MobilityHawkesModel(HawkesModelSpec(
        marks=marks, temporal_bases=2, observation="poisson"))
    result = rolling_backtest(
        model,
        data,
        horizon=4,
        step=4,
        min_train=12,
        fit_config=FitConfig(
            map_steps=20, map_ensemble_size=1, map_patience=8),
        forecast_samples=8,
        seed=10,
    )

    assert len(result.splits) >= 1
    assert not result.metrics.empty


def test_rolling_backtest_svi_method_smoke():
    grid = GridSpec(shape=(4, 4))
    marks = ("a", "b")

    params = HawkesSimParams(
        mu=0.02,
        alpha_from=jnp.array([0.15, 0.12], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )

    data = simulate_counts(
        T=16,
        grid=grid,
        num_nodes=16,
        marks=marks,
        connectivity=make_grid_stencil(size=3),
        params=params,
        seed=11,
    )

    model = MobilityHawkesModel(HawkesModelSpec(
        marks=marks, temporal_bases=2, observation="poisson"))
    result = rolling_backtest(
        model,
        data,
        horizon=4,
        step=4,
        min_train=8,
        fit_method="svi",
        fit_config=FitConfig(svi_steps=20, svi_num_particles=1),
        forecast_samples=8,
        seed=12,
    )

    assert result.meta["fit_method"] == "svi"
    assert not result.metrics.empty
