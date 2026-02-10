from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sim.world import World


@dataclass(frozen=True, slots=True)
class ChicagoData:
    """Loaded Chicago-style dataset (placeholder schema)."""

    world: World
    y_obs: np.ndarray
    meta: dict[str, object]


def load_y_obs_matrix(*, path: str | Path, mobility_path: str | Path | None = None) -> ChicagoData:
    """Load observed counts under the placeholder Chicago schema.

    Placeholder schema
    ------------------
    - `path`: CSV (no header) representing a dense matrix with shape
      (n_locations, n_steps) where rows are location indices and columns are
      daily time steps. Entries are non-negative integer counts.
    - `mobility_path` (optional): `.npy` file containing a mobility matrix of
      shape (n_locations, n_locations). If not provided, uses identity.

    Returns
    -------
    ChicagoData with fields (world, y_obs, meta).
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    y = np.loadtxt(p, delimiter=",")
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y_obs must be a 2D matrix (n_locations, n_steps)")

    # Ensure integer-valued counts.
    if not np.all(np.isfinite(y)):
        raise ValueError("y_obs must be finite")
    if np.any(y < 0):
        raise ValueError("y_obs must be non-negative")
    if not np.allclose(y, np.round(y)):
        raise ValueError("y_obs must be integer-valued")

    y_obs = np.asarray(np.round(y), dtype=int)

    n_locations, n_steps = y_obs.shape
    if n_locations <= 0 or n_steps <= 0:
        raise ValueError("y_obs must have positive shape")

    if mobility_path is None:
        mobility = np.eye(n_locations, dtype=float)
        mobility_source = "identity"
    else:
        mp = Path(mobility_path)
        if not mp.exists():
            raise FileNotFoundError(str(mp))
        mobility = np.load(mp)
        mobility = np.asarray(mobility, dtype=float)
        if mobility.shape != (n_locations, n_locations):
            raise ValueError(
                "mobility must have shape (n_locations, n_locations) matching y_obs"
            )
        if not np.all(np.isfinite(mobility)):
            raise ValueError("mobility must be finite")
        mobility_source = str(mp)

    # Placeholder xy coordinates (not used in current simulator APIs).
    world = World(xy=np.zeros((n_locations, 2), dtype=float), mobility=mobility)

    meta: dict[str, object] = {
        "schema": "placeholder-y_obs-matrix",
        "path": str(p),
        "mobility_source": mobility_source,
        "n_locations": int(n_locations),
        "n_steps": int(n_steps),
        "time_unit": "day",
    }

    return ChicagoData(world=world, y_obs=y_obs, meta=meta)
