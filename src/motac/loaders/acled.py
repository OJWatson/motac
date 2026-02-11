from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.world import World


@dataclass(frozen=True, slots=True)
class AcledData:
    """Loaded ACLED-style dataset (placeholder schema)."""

    world: World
    y_obs: np.ndarray
    meta: dict[str, object]


def load_acled_events_csv(
    *,
    path: str | Path,
    mobility_path: str | Path | None = None,
    value: str = "events",
) -> AcledData:
    """Load ACLED events CSV under a placeholder schema and aggregate to counts.

    Placeholder schema
    ------------------
    Input is a CSV (with header) containing at least:

      - event_date: YYYY-MM-DD
      - lat: float
      - lon: float
      - fatalities: integer >= 0

    Each row is an event. We aggregate to *daily* counts per location.
    Locations are defined by unique (lat, lon) pairs in the file.

    Aggregation value
    -----------------
    - value="events": count events per (location, day)
    - value="fatalities": sum fatalities per (location, day)

    Mobility
    --------
    If mobility_path is provided, load a (n_locations, n_locations) float matrix
    from `.npy`. Otherwise use identity.

    Returns
    -------
    AcledData with fields (world, y_obs, meta), where y_obs has shape
    (n_locations, n_days).
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    # Read using numpy genfromtxt to keep deps light.
    rows = np.genfromtxt(
        p,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    required = {"event_date", "lat", "lon", "fatalities"}
    names = set(rows.dtype.names or [])
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    dates = np.asarray(rows["event_date"], dtype=str)
    lat = np.asarray(rows["lat"], dtype=float)
    lon = np.asarray(rows["lon"], dtype=float)
    fatalities = np.asarray(rows["fatalities"], dtype=float)

    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("lat/lon must be finite")
    if np.any(lat < -90.0) or np.any(lat > 90.0) or np.any(lon < -180.0) or np.any(lon > 180.0):
        raise ValueError("lat/lon out of range")

    if not np.all(np.isfinite(fatalities)) or np.any(fatalities < 0.0):
        raise ValueError("fatalities must be finite and non-negative")
    if not np.allclose(fatalities, np.round(fatalities)):
        raise ValueError("fatalities must be integer-valued")

    if value not in {"events", "fatalities"}:
        raise ValueError("value must be one of {'events','fatalities'}")

    # Locations = unique (lat, lon) pairs.
    xy = np.stack([lon, lat], axis=1)
    uniq_xy, loc_idx = np.unique(xy, axis=0, return_inverse=True)

    uniq_dates = np.unique(dates)
    # Keep stable chronological order for ISO dates.
    uniq_dates = np.sort(uniq_dates)

    date_to_idx = {d: i for i, d in enumerate(uniq_dates.tolist())}
    day_idx = np.array([date_to_idx[d] for d in dates.tolist()], dtype=int)

    n_locations = int(uniq_xy.shape[0])
    n_days = int(uniq_dates.shape[0])

    y_obs = np.zeros((n_locations, n_days), dtype=int)

    if value == "events":
        np.add.at(y_obs, (loc_idx, day_idx), 1)
    else:
        np.add.at(y_obs, (loc_idx, day_idx), fatalities.astype(int))

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
                "mobility must have shape (n_locations, n_locations) matching inferred locations"
            )
        if not np.all(np.isfinite(mobility)):
            raise ValueError("mobility must be finite")
        mobility_source = str(mp)

    # World xy uses lon/lat for placeholder.
    world = World(xy=uniq_xy.astype(float), mobility=mobility)

    meta: dict[str, object] = {
        "schema": "placeholder-acled-events-csv",
        "path": str(p),
        "value": value,
        "time_unit": "day",
        "dates": uniq_dates.tolist(),
        "mobility_source": mobility_source,
        "n_locations": n_locations,
        "n_days": n_days,
        "location_xy": uniq_xy.tolist(),
    }

    return AcledData(world=world, y_obs=y_obs, meta=meta)
