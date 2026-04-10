import jax.numpy as jnp
import numpy as np

from motac.connectivity import make_grid_stencil
from motac.data import CountsTensor, GridSpec
from motac.simulate import HawkesSimNoise, HawkesSimParams, counts_to_events, simulate_counts


class DummyConnectivity:
    pass


def test_counts_to_events_expansion():
    y = jnp.array(
        [
            [[0.0, 2.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    counts = CountsTensor(
        y=y,
        t0=np.datetime64("2024-01-01"),
        dt_days=1,
        num_time=2,
        num_nodes=2,
        marks=("a", "b"),
        node_coords=None,
        grid=GridSpec(shape=(1, 2)),
        connectivity=DummyConnectivity(),
        covariates={},
    )

    events = counts_to_events(counts)
    # expected expansions: 2 + 1 + 1 = 4 rows
    assert len(events.df) == 4
    assert set(events.df["mark"].unique()) == {"a", "b"}
    assert "weight" in events.df.columns


def _toy_params() -> HawkesSimParams:
    return HawkesSimParams(
        mu=0.04,
        alpha_from=jnp.array([0.2, 0.15], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )


def test_simulate_counts_thinning_is_integer_safe():
    grid = GridSpec(shape=(4, 4))
    y_full = simulate_counts(
        T=14,
        grid=grid,
        num_nodes=16,
        marks=("a", "b"),
        connectivity=make_grid_stencil(size=3),
        params=_toy_params(),
        noise=HawkesSimNoise(thinning_p=1.0),
        seed=101,
    ).y
    y_thin = simulate_counts(
        T=14,
        grid=grid,
        num_nodes=16,
        marks=("a", "b"),
        connectivity=make_grid_stencil(size=3),
        params=_toy_params(),
        noise=HawkesSimNoise(thinning_p=0.5),
        seed=101,
    ).y

    assert y_thin.shape == y_full.shape
    assert float(y_thin.sum()) <= float(y_full.sum())
    assert np.all(np.asarray(y_thin) >= 0)
    assert np.all(np.mod(np.asarray(y_thin), 1.0) == 0.0)


def test_simulate_counts_jitter_preserves_total_without_thinning():
    grid = GridSpec(shape=(4, 4))
    y_plain = simulate_counts(
        T=10,
        grid=grid,
        num_nodes=16,
        marks=("a", "b"),
        connectivity=make_grid_stencil(size=3),
        params=_toy_params(),
        noise=HawkesSimNoise(
            thinning_p=1.0, jitter_time_p=0.0, jitter_space_p=0.0),
        seed=202,
    ).y
    y_jitter = simulate_counts(
        T=10,
        grid=grid,
        num_nodes=16,
        marks=("a", "b"),
        connectivity=make_grid_stencil(size=3),
        params=_toy_params(),
        noise=HawkesSimNoise(
            thinning_p=1.0, jitter_time_p=0.2, jitter_space_p=0.2),
        seed=202,
    ).y

    assert float(y_jitter.sum()) == float(y_plain.sum())
