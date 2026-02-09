from __future__ import annotations

import numpy as np

from motac.sim import (
    discrete_exponential_kernel,
    generate_random_world,
    sample_hawkes_observed_predictive_paths_poisson_approx,
)


def test_observed_predictive_sampling_shapes_nonneg_reproducible() -> None:
    world = generate_random_world(n_locations=3, seed=0, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=3, beta=1.0)
    mu = np.full((world.n_locations,), 0.1)
    alpha = 0.4

    y_hist = np.zeros((world.n_locations, 8), dtype=float)

    out1 = sample_hawkes_observed_predictive_paths_poisson_approx(
        world=world,
        mu=mu,
        alpha=alpha,
        kernel=kernel,
        y_history_for_intensity=y_hist,
        horizon=5,
        n_paths=4,
        seed=123,
        p_detect=0.7,
        false_rate=0.2,
    )
    out2 = sample_hawkes_observed_predictive_paths_poisson_approx(
        world=world,
        mu=mu,
        alpha=alpha,
        kernel=kernel,
        y_history_for_intensity=y_hist,
        horizon=5,
        n_paths=4,
        seed=123,
        p_detect=0.7,
        false_rate=0.2,
    )

    assert out1["y_obs"].shape == (4, world.n_locations, 5)
    assert out1["intensity_obs"].shape == (4, world.n_locations, 5)

    assert np.all(out1["y_obs"] >= 0)
    assert np.all(out1["intensity_obs"] >= 0.0)

    assert np.array_equal(out1["y_obs"], out2["y_obs"])
    assert np.allclose(out1["intensity_obs"], out2["intensity_obs"])
