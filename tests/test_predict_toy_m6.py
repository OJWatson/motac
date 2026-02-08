from __future__ import annotations

import numpy as np

from motac.sim import HawkesDiscreteParams, predict_hawkes_intensity_multi_step
from motac.sim.world import World


def test_multi_step_predict_toy_identity_mobility_lag1() -> None:
    # Toy model: 1 location, identity mobility, lag-1 kernel.
    # lambda(t) = mu + alpha * y(t-1)
    # Multi-step forecast uses expected counts roll-forward y := lambda.
    mu = np.array([1.0])
    alpha = 0.5
    kernel = np.array([1.0])

    world = World(
        xy=np.zeros((1, 2), dtype=float),
        mobility=np.array([[1.0]], dtype=float),
    )

    params = HawkesDiscreteParams(mu=mu, alpha=alpha, kernel=kernel)

    # History with last observed count y(T-1)=2.
    y_history = np.array([[2.0]])

    # Step 1: lam0 = 1 + 0.5*2 = 2
    # Step 2: lam1 = 1 + 0.5*lam0 = 2
    # In fact fixed point here is 2.
    lam = predict_hawkes_intensity_multi_step(
        world=world, params=params, y_history=y_history, horizon=3
    )

    assert lam.shape == (1, 3)
    assert np.allclose(lam[0, :], np.array([2.0, 2.0, 2.0]))
