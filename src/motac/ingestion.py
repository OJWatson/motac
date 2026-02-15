from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import EventRecord, EventTable, validate_event_record


@dataclass(frozen=True, slots=True)
class CanonicalEvents:
    """Canonical events table packaged with lightweight metadata."""

    table: pa.Table
    meta: Mapping[str, Any] | None = None


def read_raw_events_jsonl(path: str | Path) -> Iterator[EventRecord]:
    """Read raw events from newline-delimited JSON.

    Each line must be a JSON object with keys:

    Required
    --------
    - t: date string (YYYY-MM-DD)
    - lat: float
    - lon: float

    Optional
    --------
    - event_id: str
    - value: int (default: 1)
    - cell_id: int
    - mark: str
    - meta: object (will be stored as a mapping)
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise ValueError(f"invalid JSON on line {i} in {p}") from exc

            if not isinstance(obj, Mapping):
                raise ValueError(f"expected JSON object on line {i} in {p}")

            rec = EventRecord(
                event_id=obj.get("event_id"),
                t=obj.get("t", "1970-01-01"),
                lat=float(obj.get("lat")),
                lon=float(obj.get("lon")),
                cell_id=obj.get("cell_id"),
                mark=obj.get("mark"),
                value=int(obj.get("value", 1)),
                meta=obj.get("meta"),
            )
            validate_event_record(rec)
            yield rec


def ingest_records(records: Iterable[EventRecord]) -> EventTable:
    """Normalize and validate a sequence of records into an :class:`EventTable`."""

    tbl = EventTable.from_records(records)
    tbl.validate()
    return tbl


def _date32_from_datetime64_day(t: np.ndarray) -> pa.Array:
    if t.dtype.kind != "M":
        raise ValueError("t must be a datetime64 array")

    t_day = t.astype("datetime64[D]")
    days = (t_day - np.datetime64("1970-01-01", "D")).astype(np.int32)
    return pa.array(days, type=pa.date32())


def event_table_to_arrow(tbl: EventTable) -> pa.Table:
    """Convert a validated :class:`EventTable` to a canonical Arrow table."""

    tbl.validate()

    event_id = None
    if tbl.event_id is not None:
        event_id = pa.array(list(tbl.event_id), type=pa.string())

    cell_id = None
    if tbl.cell_id is not None:
        # Convert -1 sentinel to null.
        cell = np.asarray(tbl.cell_id, dtype=int)
        cell_id = pa.array([None if int(x) < 0 else int(x) for x in cell], type=pa.int32())

    mark = None
    if tbl.mark is not None:
        mark = pa.array(list(tbl.mark), type=pa.string())

    meta = None
    if tbl.meta is not None:
        meta = pa.array(
            [None if m is None else json.dumps(m, sort_keys=True) for m in tbl.meta],
            type=pa.large_string(),
        )

    cols: dict[str, pa.Array] = {
        "t": _date32_from_datetime64_day(tbl.t),
        "lat": pa.array(tbl.lat, type=pa.float64()),
        "lon": pa.array(tbl.lon, type=pa.float64()),
        "value": pa.array(tbl.value, type=pa.int64()),
    }
    if event_id is not None:
        cols["event_id"] = event_id
    if cell_id is not None:
        cols["cell_id"] = cell_id
    if mark is not None:
        cols["mark"] = mark
    if meta is not None:
        cols["meta_json"] = meta

    return pa.table(cols)


def ingest_jsonl_to_canonical_table(path: str | Path) -> pa.Table:
    """Ingest a raw JSONL event stream into the canonical Arrow table."""

    return event_table_to_arrow(ingest_records(read_raw_events_jsonl(path)))


def write_canonical_events_parquet(table: pa.Table, out_path: str | Path) -> None:
    """Write a canonical events table to a Parquet file."""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, p)


def read_canonical_events_parquet(path: str | Path) -> pa.Table:
    """Read a canonical events parquet file."""

    return pq.read_table(path)
