from __future__ import annotations

import numpy as np

from motac.sim import (
    discrete_exponential_kernel,
    generate_random_world,
    observed_fit_sample_summarize_poisson_approx,
)


def test_observed_workflow_shapes_and_reproducible() -> None:
    world = generate_random_world(n_locations=3, seed=0, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=3, beta=1.0)

    # Toy observed series.
    y_obs = np.zeros((world.n_locations, 20), dtype=int)
    y_obs[:, 5] = 2

    out1 = observed_fit_sample_summarize_poisson_approx(
        world=world,
        kernel=kernel,
        y_obs=y_obs,
        p_detect=0.7,
        false_rate=0.2,
        horizon=5,
        n_paths=10,
        seed=123,
        q=(0.1, 0.5, 0.9),
        fit_maxiter=50,
    )
    out2 = observed_fit_sample_summarize_poisson_approx(
        world=world,
        kernel=kernel,
        y_obs=y_obs,
        p_detect=0.7,
        false_rate=0.2,
        horizon=5,
        n_paths=10,
        seed=123,
        q=(0.1, 0.5, 0.9),
        fit_maxiter=50,
    )

    paths = out1["paths"]
    summary = out1["summary"]

    assert paths["y_obs"].shape == (10, world.n_locations, 5)
    assert paths["intensity_obs"].shape == (10, world.n_locations, 5)

    assert summary["mean"].shape == (world.n_locations, 5)
    assert summary["quantiles"].shape == (3, world.n_locations, 5)

    # Deterministic given the RNG seed.
    assert np.array_equal(out1["paths"]["y_obs"], out2["paths"]["y_obs"])
    assert np.allclose(out1["summary"]["mean"], out2["summary"]["mean"])
