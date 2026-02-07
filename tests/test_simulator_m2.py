from __future__ import annotations

from pathlib import Path

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    fit_hawkes_alpha_mu,
    generate_random_world,
    load_simulation_parquet,
    save_simulation_parquet,
    simulate_hawkes_counts,
)


def test_simulate_shapes_and_nonnegativity() -> None:
    world = generate_random_world(n_locations=6, seed=0, lengthscale=0.4)
    kernel = discrete_exponential_kernel(n_lags=5, beta=0.8)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.2),
        alpha=0.7,
        kernel=kernel,
        p_detect=0.9,
        false_rate=0.05,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=40, seed=1)
    y_true = out["y_true"]
    y_obs = out["y_obs"]
    intensity = out["intensity"]

    assert y_true.shape == (world.n_locations, 40)
    assert y_obs.shape == (world.n_locations, 40)
    assert intensity.shape == (world.n_locations, 40)

    assert np.all(y_true >= 0)
    assert np.all(y_obs >= 0)
    assert np.all(intensity >= 0.0)


def test_parquet_roundtrip(tmp_path: Path) -> None:
    world = generate_random_world(n_locations=4, seed=2, lengthscale=0.6)
    kernel = discrete_exponential_kernel(n_lags=4, beta=1.0)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.15),
        alpha=0.9,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=25, seed=3)
    path = tmp_path / "sim.parquet"

    save_simulation_parquet(
        path=path,
        world=world,
        params=params,
        y_true=out["y_true"],
        y_obs=out["y_obs"],
    )

    loaded = load_simulation_parquet(path=path)
    assert loaded["world"].n_locations == world.n_locations
    assert np.allclose(loaded["world"].xy, world.xy)
    assert np.allclose(loaded["world"].mobility, world.mobility)

    params2 = loaded["params"]
    assert np.allclose(params2.mu, params.mu)
    assert params2.alpha == params.alpha
    assert np.allclose(params2.kernel, params.kernel)

    assert np.array_equal(loaded["y_true"], out["y_true"])
    assert np.array_equal(loaded["y_obs"], out["y_obs"])


def test_parameter_recovery_alpha_reasonable() -> None:
    world = generate_random_world(n_locations=5, seed=10, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=6, beta=0.7)

    alpha_true = 0.8
    mu_true = np.linspace(0.05, 0.15, world.n_locations)
    params = HawkesDiscreteParams(
        mu=mu_true,
        alpha=alpha_true,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=120, seed=11)
    fit = fit_hawkes_alpha_mu(world=world, kernel=kernel, y=out["y_true"], ridge=1e-3)

    alpha_hat = float(fit["alpha"])
    mu_hat = np.asarray(fit["mu"], dtype=float)

    # Recovery is approximate; check it's in the right ballpark.
    assert 0.3 <= alpha_hat <= 1.3
    assert abs(alpha_hat - alpha_true) / alpha_true < 0.5

    # Baseline rates should be positive and roughly similar scale.
    assert mu_hat.shape == mu_true.shape
    assert np.all(mu_hat >= 0.0)
    assert np.mean(mu_hat) > 0.01
