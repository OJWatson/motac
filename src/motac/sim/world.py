from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class World:
    """A simple world definition for the M2 simulator.

    Attributes
    ----------
    xy:
        Array of shape (n_locations, 2) giving planar coordinates.
    mobility:
        Influence / mobility kernel of shape (n_locations, n_locations).
        Entry mobility[i, j] controls how much activity at j contributes to i.
    """

    xy: np.ndarray
    mobility: np.ndarray

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            msg = f"xy must have shape (n,2); got {self.xy.shape}"
            raise ValueError(msg)
        if self.mobility.ndim != 2 or self.mobility.shape[0] != self.mobility.shape[1]:
            msg = f"mobility must be square; got {self.mobility.shape}"
            raise ValueError(msg)
        if self.mobility.shape[0] != self.xy.shape[0]:
            msg = (
                "mobility and xy must agree on n_locations; "
                f"got {self.mobility.shape[0]} and {self.xy.shape[0]}"
            )
            raise ValueError(msg)

    @property
    def n_locations(self) -> int:
        return int(self.xy.shape[0])

    def to_json(self) -> str:
        payload = {
            "xy": self.xy.tolist(),
            "mobility": self.mobility.tolist(),
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(text: str) -> World:
        payload = json.loads(text)
        xy = np.asarray(payload["xy"], dtype=float)
        mobility = np.asarray(payload["mobility"], dtype=float)
        return World(xy=xy, mobility=mobility)


def generate_random_world(
    *,
    n_locations: int,
    seed: int,
    lengthscale: float = 1.0,
    self_weight: float = 1.0,
    row_normalize: bool = True,
) -> World:
    """Generate a random world with an RBF mobility kernel.

    This is intentionally minimal and dependency-free; it is used primarily
    for tests and examples.
    """

    if n_locations <= 0:
        raise ValueError("n_locations must be positive")
    if lengthscale <= 0:
        raise ValueError("lengthscale must be positive")

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 1.0, size=(n_locations, 2))

    # Squared distances.
    diff = xy[:, None, :] - xy[None, :, :]
    d2 = np.sum(diff**2, axis=-1)

    mobility = np.exp(-0.5 * d2 / (lengthscale**2))
    # Encourage some self-excitation when desired.
    np.fill_diagonal(mobility, np.diag(mobility) * float(self_weight))

    if row_normalize:
        row_sum = mobility.sum(axis=1, keepdims=True)
        mobility = mobility / np.where(row_sum == 0.0, 1.0, row_sum)

    return World(xy=xy, mobility=mobility)
