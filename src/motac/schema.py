from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Canonical, dataset-agnostic event record.

    Notes
    -----
    This is the smallest stable record schema we commit to across datasets.

    Attributes
    ----------
    t:
        Event time in seconds (float). Use a consistent epoch per dataset.
    lat, lon:
        WGS84 latitude/longitude in decimal degrees.
    mark:
        Optional discrete mark (e.g. event type).
    meta:
        Optional metadata (free-form, dataset specific).
    """

    t: float
    lat: float
    lon: float
    mark: str | None = None
    meta: Mapping[str, Any] | None = None


# Backwards-compatible alias (older docs/tests may refer to `Event`).
Event = EventRecord


@dataclass(frozen=True, slots=True)
class EventTable:
    """Columnar representation of many events.

    This is a convenient, validation-friendly format for loaders and downstream
    transformations.
    """

    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    mark: Sequence[str | None] | None = None
    meta: Sequence[Mapping[str, Any] | None] | None = None

    @property
    def n_events(self) -> int:
        return int(self.t.shape[0])

    @staticmethod
    def from_records(records: Iterable[EventRecord]) -> EventTable:
        recs = list(records)
        t = np.asarray([r.t for r in recs], dtype=float)
        lat = np.asarray([r.lat for r in recs], dtype=float)
        lon = np.asarray([r.lon for r in recs], dtype=float)
        mark = [r.mark for r in recs]
        meta = [r.meta for r in recs]
        return EventTable(t=t, lat=lat, lon=lon, mark=mark, meta=meta)

    def to_records(self) -> list[EventRecord]:
        mark = self.mark
        meta = self.meta
        out: list[EventRecord] = []
        for i in range(self.n_events):
            out.append(
                EventRecord(
                    t=float(self.t[i]),
                    lat=float(self.lat[i]),
                    lon=float(self.lon[i]),
                    mark=None if mark is None else mark[i],
                    meta=None if meta is None else meta[i],
                )
            )
        return out


def validate_event_record(e: EventRecord) -> None:
    """Raise ``ValueError`` if an event record is invalid."""

    for name, v in ("t", e.t), ("lat", e.lat), ("lon", e.lon):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")

    if not (-90.0 <= float(e.lat) <= 90.0):
        raise ValueError(f"lat out of range: {e.lat!r}")
    if not (-180.0 <= float(e.lon) <= 180.0):
        raise ValueError(f"lon out of range: {e.lon!r}")

    if e.mark is not None and (not isinstance(e.mark, str) or e.mark == ""):
        raise ValueError("mark must be a non-empty string or None")

    if e.meta is not None and not isinstance(e.meta, Mapping):
        raise ValueError("meta must be a Mapping or None")


def validate_event_table(tbl: EventTable) -> None:
    """Raise ``ValueError`` if an event table is invalid."""

    for name, a in ("t", tbl.t), ("lat", tbl.lat), ("lon", tbl.lon):
        if not isinstance(a, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")

    n = tbl.t.shape[0]
    if tbl.lat.shape[0] != n or tbl.lon.shape[0] != n:
        raise ValueError("t/lat/lon must have the same length")

    if np.any(tbl.lat < -90.0) or np.any(tbl.lat > 90.0):
        raise ValueError("lat out of range")
    if np.any(tbl.lon < -180.0) or np.any(tbl.lon > 180.0):
        raise ValueError("lon out of range")

    if tbl.mark is not None and len(tbl.mark) != n:
        raise ValueError("mark must have length n_events")
    if tbl.meta is not None and len(tbl.meta) != n:
        raise ValueError("meta must have length n_events")

    # Spot-check mark/meta types if present.
    if tbl.mark is not None:
        for m in tbl.mark:
            if m is not None and (not isinstance(m, str) or m == ""):
                raise ValueError("mark entries must be non-empty strings or None")
    if tbl.meta is not None:
        for m in tbl.meta:
            if m is not None and not isinstance(m, Mapping):
                raise ValueError("meta entries must be Mappings or None")
