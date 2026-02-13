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

    Notes
    -----
    - Time is represented at **day resolution**.
    - Some workflows assign events into a spatial discretisation (e.g. a grid).
      When available, loaders can populate ``cell_id``.

    Attributes
    ----------
    event_id:
        Optional stable identifier (dataset-specific). When absent, downstream
        code should treat (t, lat, lon, mark, value, meta) as the record.
    t:
        Event date (YYYY-MM-DD) or `np.datetime64`. Values are normalized to
        day resolution (`datetime64[D]`).
    lat, lon:
        WGS84 latitude/longitude in decimal degrees.
    cell_id:
        Optional non-negative integer id for a spatial cell.
    mark:
        Optional discrete mark (e.g. event type).
    value:
        Non-negative integer value (e.g. 1 per event, or fatalities).
    meta:
        Optional metadata (free-form, dataset specific).
    """

    event_id: str | None = None
    t: str | np.datetime64 = "1970-01-01"
    lat: float = 0.0
    lon: float = 0.0
    cell_id: int | None = None
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

    Optional fields (event_id, cell_id, mark, meta) are stored as Python
    sequences to keep the schema lightweight and loader-friendly.
    """

    t: np.ndarray  # datetime64[D]
    lat: np.ndarray
    lon: np.ndarray
    value: np.ndarray
    event_id: Sequence[str | None] | None = None
    cell_id: np.ndarray | None = None
    mark: Sequence[str | None] | None = None
    meta: Sequence[Mapping[str, Any] | None] | None = None

    @property
    def n_events(self) -> int:
        return int(self.t.shape[0])

    @staticmethod
    def from_records(records: Iterable[EventRecord]) -> EventTable:
        recs = list(records)
        if len(recs) == 0:
            raise ValueError("records must be non-empty")

        t = np.asarray([_as_datetime64_day(r.t) for r in recs], dtype="datetime64[D]")
        lat = np.asarray([r.lat for r in recs], dtype=float)
        lon = np.asarray([r.lon for r in recs], dtype=float)
        value = np.asarray([r.value for r in recs], dtype=int)
        event_id_list = [r.event_id for r in recs]
        event_id = None if all(x is None for x in event_id_list) else event_id_list

        cell_id_list = [r.cell_id for r in recs]
        cell_id = None
        if not all(x is None for x in cell_id_list):
            # Use -1 as a sentinel for "unknown" to keep an integer array.
            cell_id = np.asarray([(-1 if x is None else int(x)) for x in cell_id_list], dtype=int)
        mark = [r.mark for r in recs]
        meta = [r.meta for r in recs]

        return EventTable(
            t=t,
            lat=lat,
            lon=lon,
            value=value,
            event_id=event_id,
            cell_id=cell_id,
            mark=mark,
            meta=meta,
        )

    def to_records(self) -> list[EventRecord]:
        event_id = self.event_id
        cell_id = self.cell_id
        mark = self.mark
        meta = self.meta

        out: list[EventRecord] = []
        for i in range(self.n_events):
            cid: int | None = None
            if cell_id is not None:
                cid_val = int(cell_id[i])
                cid = None if cid_val < 0 else cid_val

            out.append(
                EventRecord(
                    event_id=None if event_id is None else event_id[i],
                    t=self.t[i],
                    lat=float(self.lat[i]),
                    lon=float(self.lon[i]),
                    cell_id=cid,
                    mark=None if mark is None else mark[i],
                    value=int(self.value[i]),
                    meta=None if meta is None else meta[i],
                )
            )
        return out

    def validate(self) -> None:
        """Validate this table (raises ValueError if invalid)."""

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

    if e.event_id is not None and (not isinstance(e.event_id, str) or e.event_id == ""):
        raise ValueError("event_id must be a non-empty string or None")

    if e.cell_id is not None and (not isinstance(e.cell_id, int) or e.cell_id < 0):
        raise ValueError("cell_id must be a non-negative int or None")

    if e.mark is not None and (not isinstance(e.mark, str) or e.mark == ""):
        raise ValueError("mark must be a non-empty string or None")

    if e.meta is not None and not isinstance(e.meta, Mapping):
        raise ValueError("meta must be a Mapping or None")


def validate_event_table(tbl: EventTable) -> None:
    """Raise ``ValueError`` if an event table is invalid."""

    for name, a in (
        ("t", tbl.t),
        ("lat", tbl.lat),
        ("lon", tbl.lon),
        ("value", tbl.value),
    ):
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

    if tbl.event_id is not None and len(tbl.event_id) != n:
        raise ValueError("event_id must have length n_events")
    if tbl.cell_id is not None:
        if not isinstance(tbl.cell_id, np.ndarray):
            raise ValueError("cell_id must be a numpy array")
        if tbl.cell_id.shape != (n,):
            raise ValueError("cell_id must have shape (n_events,)")
        # -1 sentinel is allowed for unknown cell assignment.
        if np.any(tbl.cell_id < -1):
            raise ValueError("cell_id must be >= -1")

    if tbl.mark is not None and len(tbl.mark) != n:
        raise ValueError("mark must have length n_events")
    if tbl.meta is not None and len(tbl.meta) != n:
        raise ValueError("meta must have length n_events")

    if tbl.event_id is not None:
        for eid in tbl.event_id:
            if eid is not None and (not isinstance(eid, str) or eid == ""):
                raise ValueError("event_id entries must be non-empty strings or None")

    if tbl.mark is not None:
        for m in tbl.mark:
            if m is not None and (not isinstance(m, str) or m == ""):
                raise ValueError("mark entries must be non-empty strings or None")

    if tbl.meta is not None:
        for m in tbl.meta:
            if m is not None and not isinstance(m, Mapping):
                raise ValueError("meta entries must be Mappings or None")
