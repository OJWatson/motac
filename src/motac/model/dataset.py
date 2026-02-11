from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motac.substrate.types import Substrate


@dataclass(frozen=True, slots=True)
class RoadHawkesDataset:
    """Dataset wiring for the road-constrained Hawkes count model.

    This is a thin container that ties together:
    - a road-constrained substrate (grid + sparse travel-time neighbours)
    - an observed count matrix y_obs with shape (n_cells, n_steps)

    The parametric model uses `substrate.neighbours.travel_time_s` as the sparse
    neighbourhood graph.
    """

    substrate: Substrate
    y_obs: np.ndarray

    def __post_init__(self) -> None:
        y = np.asarray(self.y_obs)
        if y.ndim != 2:
            raise ValueError("y_obs must be 2D (n_cells, n_steps)")

        n_cells = int(self.substrate.neighbours.travel_time_s.shape[0])
        if y.shape[0] != n_cells:
            raise ValueError("y_obs first dimension must match substrate n_cells")

        if np.any(y < 0):
            raise ValueError("y_obs must be non-negative")

    @property
    def n_cells(self) -> int:
        return int(self.y_obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.y_obs.shape[1])
