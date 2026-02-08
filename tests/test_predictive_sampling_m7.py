from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    sample_hawkes_predictive_paths,
)


def test_predictive_sampling_shapes_nonneg_and_reproducible() -> None:
    world = generate_random_world(n_locations=4, seed=0, lengthscale=0.5)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.1),
        alpha=0.6,
        kernel=discrete_exponential_kernel(n_lags=4, beta=1.0),
        p_detect=0.7,
        false_rate=0.2,
    )

    y_hist = np.zeros((world.n_locations, 10), dtype=int)

    out1 = sample_hawkes_predictive_paths(
        world=world,
        params=params,
        y_history=y_hist,
        horizon=6,
        n_paths=3,
        seed=123,
    )
    out2 = sample_hawkes_predictive_paths(
        world=world,
        params=params,
        y_history=y_hist,
        horizon=6,
        n_paths=3,
        seed=123,
    )

    assert out1["y_true"].shape == (3, world.n_locations, 6)
    assert out1["y_obs"].shape == (3, world.n_locations, 6)
    assert out1["intensity"].shape == (3, world.n_locations, 6)

    assert np.all(out1["y_true"] >= 0)
    assert np.all(out1["y_obs"] >= 0)
    assert np.all(out1["intensity"] >= 0.0)

    # Reproducible with same seed.
    assert np.array_equal(out1["y_true"], out2["y_true"])
    assert np.array_equal(out1["y_obs"], out2["y_obs"])
    assert np.allclose(out1["intensity"], out2["intensity"])
