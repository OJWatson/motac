from __future__ import annotations

import numpy as np

from motac.sim import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    generate_random_world,
    predict_hawkes_intensity_multi_step,
    predict_hawkes_intensity_one_step,
    simulate_hawkes_counts,
)


def test_predict_one_step_matches_simulator_intensity() -> None:
    world = generate_random_world(n_locations=5, seed=123, lengthscale=0.5)
    kernel = discrete_exponential_kernel(n_lags=4, beta=0.8)
    params = HawkesDiscreteParams(
        mu=np.linspace(0.05, 0.15, world.n_locations),
        alpha=0.9,
        kernel=kernel,
        p_detect=1.0,
        false_rate=0.0,
    )

    out = simulate_hawkes_counts(world=world, params=params, n_steps=30, seed=7)
    y = out["y_true"]
    intensity = out["intensity"]

    # For each t, the simulator's intensity[:, t] is computed from history y[:, :t].
    for t in range(y.shape[1]):
        lam_hat = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y[:, :t])
        assert lam_hat.shape == (world.n_locations,)
        assert np.allclose(lam_hat, intensity[:, t])


def test_predict_multi_step_shapes_and_consistency() -> None:
    world = generate_random_world(n_locations=4, seed=0, lengthscale=0.6)
    kernel = discrete_exponential_kernel(n_lags=3, beta=1.1)
    params = HawkesDiscreteParams(
        mu=np.full((world.n_locations,), 0.2),
        alpha=0.7,
        kernel=kernel,
    )

    y_hist = np.zeros((world.n_locations, 5), dtype=int)
    horizon = 6

    lam_multi = predict_hawkes_intensity_multi_step(
        world=world, params=params, y_history=y_hist, horizon=horizon
    )

    assert lam_multi.shape == (world.n_locations, horizon)
    assert np.all(lam_multi >= 0.0)

    lam_one = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y_hist)
    assert np.allclose(lam_multi[:, 0], lam_one)


def test_predict_alpha_zero_reduces_to_mu() -> None:
    world = generate_random_world(n_locations=3, seed=5, lengthscale=0.4)
    kernel = discrete_exponential_kernel(n_lags=5, beta=0.9)
    mu = np.array([0.1, 0.2, 0.3])
    params = HawkesDiscreteParams(mu=mu, alpha=0.0, kernel=kernel)

    y_hist = np.random.default_rng(0).poisson(lam=2.0, size=(world.n_locations, 20))

    lam_one = predict_hawkes_intensity_one_step(world=world, params=params, y_history=y_hist)
    assert np.allclose(lam_one, mu)

    lam_multi = predict_hawkes_intensity_multi_step(
        world=world, params=params, y_history=y_hist, horizon=4
    )
    assert np.allclose(lam_multi, mu.reshape(-1, 1))
