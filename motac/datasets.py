"""Dataset adapters for Chicago and ACLED.

Network calls are intentionally optional and should be avoided in test suites.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data import CountsTensor, EventTable, GridSpec


def fetch_chicago_crimes(
    start: date,
    end: date,
    *,
    where: str | None = None,
    limit: int = 200_000,
    app_token: str | None = None,
    use_sodapy: bool = True,
    cache_dir: Path | None = None,
) -> EventTable:
    """Fetch Chicago crime events and return a normalized :class:`EventTable`.

    Parameters
    ----------
    start, end:
        Inclusive date bounds for the query window.
    where:
        Optional Socrata filter expression appended to the default constraints.
    limit:
        Maximum number of records requested from the API.
    app_token:
        Optional Socrata app token for higher request quotas.
    use_sodapy:
        If ``True``, use the ``sodapy`` client; otherwise perform a direct HTTP query.
    cache_dir:
        Optional directory for CSV cache read/write.

    Returns
    -------
    EventTable
        Event table with canonical Chicago column mapping.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / \
            f"chicago_{start.isoformat()}_{end.isoformat()}.csv"
        if cache_file.exists():
            return EventTable(
                df=pd.read_csv(cache_file),
                time_col="date",
                lat_col="latitude",
                lon_col="longitude",
                mark_col="primary_type",
            )

    if use_sodapy:
        try:
            from sodapy import Socrata
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "sodapy is required when use_sodapy=True") from exc

        client = Socrata("data.cityofchicago.org", app_token)
        where_parts = [
            f"date >= '{start.isoformat()}T00:00:00'",
            f"date <= '{end.isoformat()}T23:59:59'",
            "latitude IS NOT NULL",
            "longitude IS NOT NULL",
        ]
        if where:
            where_parts.append(f"({where})")
        where_clause = " AND ".join(where_parts)
        rows = client.get("ijzp-q8t2", where=where_clause, limit=limit)
        df = pd.DataFrame.from_records(rows)
    else:
        where_parts = [
            f"date >= '{start.isoformat()}T00:00:00'",
            f"date <= '{end.isoformat()}T23:59:59'",
            "latitude IS NOT NULL",
            "longitude IS NOT NULL",
        ]
        if where:
            where_parts.append(f"({where})")
        where_clause = " AND ".join(where_parts)
        query = urlencode({"$where": where_clause, "$limit": limit})
        url = f"https://data.cityofchicago.org/resource/ijzp-q8t2.json?{query}"
        with urlopen(url) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        df = pd.DataFrame.from_records(payload)

    if cache_dir is not None:
        cache_file = cache_dir / \
            f"chicago_{start.isoformat()}_{end.isoformat()}.csv"
        df.to_csv(cache_file, index=False)

    return EventTable(
        df=df,
        time_col="date",
        lat_col="latitude",
        lon_col="longitude",
        mark_col="primary_type",
    )


def fetch_acled_gaza(
    start: date,
    end: date,
    *,
    mode: str = "full",
    region: str = "gaza",
    cache_dir: Path | None = None,
) -> EventTable:
    """Fetch ACLED events through ``acled_viz`` and return an :class:`EventTable`.

    This adapter is intentionally cache-first for reproducibility and to reduce
    external dependency on live API availability.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / \
            f"acled_{region}_{start.isoformat()}_{end.isoformat()}.csv"
        if cache_file.exists():
            return EventTable(
                df=pd.read_csv(cache_file),
                time_col="event_date",
                lat_col="latitude",
                lon_col="longitude",
                mark_col="event_type",
            )

    try:
        from acled_viz.data_fetcher import fetch_events as acled_fetch_events
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "acled_viz is required for live ACLED fetch. Use local snapshot CSV in cache_dir otherwise."
        ) from exc

    df = acled_fetch_events(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        mode=mode,
        region=region,
    )
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "acled_viz fetch_events returned unexpected object type")

    if cache_dir is not None:
        cache_file = cache_dir / \
            f"acled_{region}_{start.isoformat()}_{end.isoformat()}.csv"
        df.to_csv(cache_file, index=False)

    return EventTable(
        df=df,
        time_col="event_date",
        lat_col="latitude",
        lon_col="longitude",
        mark_col="event_type",
    )


def _events_to_counts(
    events: EventTable,
    grid: GridSpec,
    marks: Sequence[str] | None,
    *,
    dt_days: int,
    mark_mapping: dict[str, str] | None = None,
    drop_na_locations: bool = True,
) -> CountsTensor:
    """Convert an :class:`EventTable` into a canonical `[T, J, M]` count tensor.

    Notes
    -----
    - Time is discretized into fixed-width bins (`dt_days`).
    - Events outside known mark set are ignored after mapping.
    - Spatial assignment is performed by clipping into the provided grid extent.
    """
    df = events.df.copy()
    events.validate()

    if drop_na_locations:
        df = df.dropna(subset=[events.lat_col, events.lon_col])

    if mark_mapping is not None:
        df[events.mark_col] = df[events.mark_col].map(
            lambda v: mark_mapping.get(v, v))

    if len(df) == 0:
        raise ValueError(
            "No events available after filtering; cannot build CountsTensor")

    if marks is None:
        marks = tuple(
            sorted(df[events.mark_col].dropna().astype(str).unique()))
    else:
        marks = tuple(marks)

    t_series = pd.to_datetime(
        df[events.time_col], utc=True, errors="coerce").dt.floor("D")
    valid_time = t_series.notna()
    if not valid_time.any():
        raise ValueError("All event timestamps are invalid/NaT after parsing")
    df = df.loc[valid_time].copy()
    t_series = t_series.loc[valid_time]
    t0 = np.datetime64(t_series.min().date())
    tmax = np.datetime64(t_series.max().date())
    num_time = int(((tmax - t0) / np.timedelta64(dt_days, "D")) + 1)

    lon_min, lon_max, lat_min, lat_max = (
        grid.extent if grid.extent is not None else (
            -180.0, 180.0, -90.0, 90.0)
    )
    h, w = grid.shape

    lon_edges = np.linspace(lon_min, lon_max, w + 1, dtype=np.float32)
    lat_edges = np.linspace(lat_min, lat_max, h + 1, dtype=np.float32)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    grid_xx, grid_yy = np.meshgrid(lon_centers, lat_centers, indexing="xy")
    node_coords = np.column_stack(
        [grid_xx.reshape(-1), grid_yy.reshape(-1)]).astype(np.float32, copy=False)

    lon = df[events.lon_col].to_numpy(dtype=float)
    lat = df[events.lat_col].to_numpy(dtype=float)
    rr = np.clip(((lat - lat_min) / max(lat_max - lat_min, 1e-8)
                 * h).astype(int), 0, h - 1)
    cc = np.clip(((lon - lon_min) / max(lon_max - lon_min, 1e-8)
                 * w).astype(int), 0, w - 1)
    node = rr * w + cc

    mark_index = {m: i for i, m in enumerate(marks)}
    mark_id = df[events.mark_col].astype(str).map(
        mark_index).fillna(-1).astype(int).to_numpy()

    time_id = ((t_series.to_numpy(
        dtype="datetime64[D]") - t0) / np.timedelta64(dt_days, "D")).astype(int)
    y = np.zeros((num_time, h * w, len(marks)), dtype=np.float32)

    valid = mark_id >= 0
    if np.any(valid):
        np.add.at(
            y,
            (time_id[valid], node[valid], mark_id[valid]),
            1.0,
        )

    return CountsTensor(
        y=jnp.asarray(y),
        t0=t0,
        dt_days=dt_days,
        num_time=num_time,
        num_nodes=h * w,
        marks=marks,
        node_coords=node_coords,
        grid=grid,
        connectivity=None,
        covariates={},
    )


def chicago_to_counts(
    events: EventTable,
    grid: GridSpec,
    marks: Sequence[str] | None = None,
    *,
    dt_days: int = 1,
    drop_na_locations: bool = True,
) -> CountsTensor:
    """Convert Chicago-style event tables into `CountsTensor`.

    This is a thin wrapper over :func:`_events_to_counts` with no mark remapping.
    """
    return _events_to_counts(
        events,
        grid,
        marks,
        dt_days=dt_days,
        mark_mapping=None,
        drop_na_locations=drop_na_locations,
    )


def acled_to_counts(
    events: EventTable,
    grid: GridSpec,
    marks: Sequence[str] | None = None,
    *,
    dt_days: int = 1,
    mark_mapping: dict[str, str] | None = None,
) -> CountsTensor:
    """Convert ACLED-style event tables into `CountsTensor`.

    Parameters are identical to :func:`_events_to_counts`, with optional
    `mark_mapping` for taxonomy harmonization.
    """
    return _events_to_counts(
        events,
        grid,
        marks,
        dt_days=dt_days,
        mark_mapping=mark_mapping,
        drop_na_locations=True,
    )
