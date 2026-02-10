from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim.world import World


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


@dataclass(frozen=True, slots=True)
class NeuralIntensityBaseline:
    """Deterministic baseline "neural" intensity model.

    This is a placeholder for M7 (Neural). It intentionally avoids heavy
    dependencies and training loops.

    Model
    -----
    We build a simple feature from recent history:

        f_i = mean_t(y_i[t-k:t])

    and predict

        lambda_i = softplus(bias_i + w * f_i)

    Parameters
    ----------
    bias:
        Per-location bias term, shape (n_locations,).
    weight:
        Global positive weight for recent activity.
    window:
        Number of recent steps used in the feature.
    """

    bias: np.ndarray
    weight: float = 1.0
    window: int = 7

    def __post_init__(self) -> None:
        if self.bias.ndim != 1:
            raise ValueError("bias must be 1D")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")

    def predict_intensity(self, *, world: World, y_history: np.ndarray) -> np.ndarray:
        """One-step intensity prediction given history.

        Parameters
        ----------
        world:
            Provides n_locations.
        y_history:
            Past counts, shape (n_locations, n_steps).

        Returns
        -------
        lam_next:
            Non-negative intensities, shape (n_locations,).
        """

        if y_history.ndim != 2:
            raise ValueError("y_history must be 2D")
        if y_history.shape[0] != world.n_locations:
            raise ValueError("y_history first dimension must match world.n_locations")
        if self.bias.shape != (world.n_locations,):
            raise ValueError("bias must have shape (n_locations,)")

        t = int(y_history.shape[1])
        start = max(0, t - int(self.window))
        recent = np.asarray(y_history[:, start:t], dtype=float)
        feat = recent.mean(axis=1) if recent.size > 0 else np.zeros((world.n_locations,))

        x = self.bias + float(self.weight) * feat
        lam = _softplus(x)
        return np.clip(lam, 0.0, None)
