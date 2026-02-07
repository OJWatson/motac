from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .hawkes import HawkesDiscreteParams
from .world import World


def _stack_long(
    *,
    y_true: np.ndarray,
    y_obs: np.ndarray,
) -> dict[str, Any]:
    n_locations, n_steps = y_true.shape
    loc = np.repeat(np.arange(n_locations, dtype=int), n_steps)
    time = np.tile(np.arange(n_steps, dtype=int), n_locations)
    return {
        "location": loc,
        "time": time,
        "y_true": y_true.reshape(-1).astype(np.int64),
        "y_obs": y_obs.reshape(-1).astype(np.int64),
    }


def save_simulation_parquet(
    *,
    path: str | Path,
    world: World,
    params: HawkesDiscreteParams,
    y_true: np.ndarray,
    y_obs: np.ndarray,
) -> None:
    """Save simulation output to a Parquet file.

    This uses PyArrow directly to avoid pulling in pandas.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if y_true.shape != y_obs.shape:
        raise ValueError("y_true and y_obs must have the same shape")

    data = _stack_long(y_true=y_true, y_obs=y_obs)
    table = pa.table(data)

    meta = {
        b"motac:world": world.to_json().encode("utf-8"),
        b"motac:params": params.to_json().encode("utf-8"),
        b"motac:schema": json.dumps({"format": "long-v1"}).encode("utf-8"),
    }

    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, path)


def load_simulation_parquet(
    *,
    path: str | Path,
) -> dict[str, Any]:
    """Load simulation outputs saved with :func:`save_simulation_parquet`."""

    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(path)
    meta = table.schema.metadata or {}

    if b"motac:world" not in meta or b"motac:params" not in meta:
        raise ValueError("Parquet file missing required motac metadata")

    world = World.from_json(meta[b"motac:world"].decode("utf-8"))
    params = HawkesDiscreteParams.from_json(meta[b"motac:params"].decode("utf-8"))

    df = table.to_pydict()
    location = np.asarray(df["location"], dtype=int)
    time = np.asarray(df["time"], dtype=int)
    y_true = np.asarray(df["y_true"], dtype=np.int64)
    y_obs = np.asarray(df["y_obs"], dtype=np.int64)

    n_locations = int(location.max() + 1) if location.size else 0
    n_steps = int(time.max() + 1) if time.size else 0

    y_true_mat = np.zeros((n_locations, n_steps), dtype=np.int64)
    y_obs_mat = np.zeros((n_locations, n_steps), dtype=np.int64)
    y_true_mat[location, time] = y_true
    y_obs_mat[location, time] = y_obs

    return {
        "world": world,
        "params": params,
        "y_true": y_true_mat,
        "y_obs": y_obs_mat,
    }
