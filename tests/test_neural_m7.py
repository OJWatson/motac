from __future__ import annotations

import numpy as np

from motac.neural import NeuralIntensityBaseline
from motac.sim.world import World


def test_neural_baseline_shapes_nonneg_reproducible() -> None:
    world = World(xy=np.zeros((3, 2), dtype=float), mobility=np.eye(3))
    model = NeuralIntensityBaseline(bias=np.array([0.0, -1.0, 0.5]), weight=2.0, window=3)

    y_hist = np.zeros((3, 5), dtype=int)
    y_hist[1, -1] = 4

    lam1 = model.predict_intensity(world=world, y_history=y_hist)
    lam2 = model.predict_intensity(world=world, y_history=y_hist)

    assert lam1.shape == (3,)
    assert np.all(lam1 >= 0.0)
    assert np.allclose(lam1, lam2)


def test_neural_baseline_monotonic_recent_counts() -> None:
    world = World(xy=np.zeros((2, 2), dtype=float), mobility=np.eye(2))
    model = NeuralIntensityBaseline(bias=np.array([0.0, 0.0]), weight=1.0, window=2)

    y0 = np.zeros((2, 4), dtype=int)
    y1 = y0.copy()
    y1[0, -1] = 10

    lam0 = model.predict_intensity(world=world, y_history=y0)
    lam1 = model.predict_intensity(world=world, y_history=y1)

    assert lam1[0] > lam0[0]
    assert np.allclose(lam1[1], lam0[1])
