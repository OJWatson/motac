from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_datetime64_day(x: str | np.datetime64) -> np.datetime64:
    """Parse a date-like value into numpy datetime64[D]."""

    return np.datetime64(x, "D")


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Canonical, dataset-agnostic event record.

    This is the canonical schema representation used by dataset loaders.

    Attributes
    ----------
    t:
        Event date (YYYY-MM-DD) or `np.datetime64`. Values are normalized to
        day resolution (`datetime64[D]`).
    lat, lon:
        WGS84 latitude/longitude in decimal degrees.
    mark:
        Optional discrete mark (e.g. event type).
    value:
        Non-negative integer value (e.g. 1 per event, or fatalities).
    meta:
        Optional metadata (free-form, dataset specific).
    """

    t: str | np.datetime64
    lat: float
    lon: float
    mark: str | None = None
    value: int = 1
    meta: Mapping[str, Any] | None = None


# Backwards-compatible alias (older docs/tests may refer to `Event`).
Event = EventRecord


@dataclass(frozen=True, slots=True)
class EventTable:
    """Columnar representation of many events.

    Times are stored as `datetime64[D]` and values are stored as non-negative
    integers.
    """

    t: np.ndarray  # datetime64[D]
    lat: np.ndarray
    lon: np.ndarray
    value: np.ndarray
    mark: Sequence[str | None] | None = None
    meta: Sequence[Mapping[str, Any] | None] | None = None

    @property
    def n_events(self) -> int:
        return int(self.t.shape[0])

    @staticmethod
    def from_records(records: Iterable[EventRecord]) -> EventTable:
        recs = list(records)
        t = np.asarray([_as_datetime64_day(r.t) for r in recs], dtype="datetime64[D]")
        lat = np.asarray([r.lat for r in recs], dtype=float)
        lon = np.asarray([r.lon for r in recs], dtype=float)
        value = np.asarray([r.value for r in recs], dtype=int)
        mark = [r.mark for r in recs]
        meta = [r.meta for r in recs]
        return EventTable(t=t, lat=lat, lon=lon, value=value, mark=mark, meta=meta)

    def to_records(self) -> list[EventRecord]:
        mark = self.mark
        meta = self.meta
        out: list[EventRecord] = []
        for i in range(self.n_events):
            out.append(
                EventRecord(
                    t=self.t[i],
                    lat=float(self.lat[i]),
                    lon=float(self.lon[i]),
                    mark=None if mark is None else mark[i],
                    value=int(self.value[i]),
                    meta=None if meta is None else meta[i],
                )
            )
        return out

    def validate(self) -> None:
        """Validate this table (raises ValueError if invalid)."""

        validate_event_table(self)

    def validate(self) -> None:
        """Validate this table.

        Raises
        ------
        ValueError
            If the table is invalid.
        """

        validate_event_table(self)


def validate_event_record(e: EventRecord) -> None:
    """Raise ``ValueError`` if an event record is invalid."""

    # Date parsing.
    try:
        _as_datetime64_day(e.t)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"t must be YYYY-MM-DD or datetime64, got {e.t!r}") from exc

    if not np.isfinite(e.lat) or not np.isfinite(e.lon):
        raise ValueError("lat/lon must be finite")

    if not (-90.0 <= float(e.lat) <= 90.0):
        raise ValueError(f"lat out of range: {e.lat!r}")
    if not (-180.0 <= float(e.lon) <= 180.0):
        raise ValueError(f"lon out of range: {e.lon!r}")

    if e.value < 0:
        raise ValueError("value must be non-negative")

    if e.mark is not None and (not isinstance(e.mark, str) or e.mark == ""):
        raise ValueError("mark must be a non-empty string or None")

    if e.meta is not None and not isinstance(e.meta, Mapping):
        raise ValueError("meta must be a Mapping or None")


def validate_event_table(tbl: EventTable) -> None:
    """Raise ``ValueError`` if an event table is invalid."""

    for name, a in ("t", tbl.t), ("lat", tbl.lat), ("lon", tbl.lon), ("value", tbl.value):
        if not isinstance(a, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape {a.shape}")

    if tbl.t.dtype.kind != "M":
        raise ValueError("t must be datetime64")

    for name, a in ("lat", tbl.lat), ("lon", tbl.lon):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")

    n = tbl.t.shape[0]
    if tbl.lat.shape[0] != n or tbl.lon.shape[0] != n or tbl.value.shape[0] != n:
        raise ValueError("t/lat/lon/value must have the same length")

    if np.any(tbl.lat < -90.0) or np.any(tbl.lat > 90.0):
        raise ValueError("lat out of range")
    if np.any(tbl.lon < -180.0) or np.any(tbl.lon > 180.0):
        raise ValueError("lon out of range")

    if np.any(tbl.value < 0):
        raise ValueError("value must be non-negative")

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
