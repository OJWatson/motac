from __future__ import annotations

import json

import pyarrow as pa

from motac.ingestion import (
    ingest_jsonl_to_canonical_table,
    read_canonical_events_parquet,
    validate_canonical_events_table,
    write_canonical_events_parquet,
)


def test_ingest_jsonl_to_canonical_table(tmp_path) -> None:
    p = tmp_path / "events.jsonl"
    rows = [
        {"t": "2020-01-02", "lat": 51.5, "lon": -0.1, "value": 2, "event_id": "a"},
        {"t": "2020-01-03", "lat": 52.0, "lon": 0.1, "mark": "x", "meta": {"k": 1}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    tbl = ingest_jsonl_to_canonical_table(p)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == 2
    validate_canonical_events_table(tbl)
    assert tbl.schema.field("t").type == pa.date32()


def test_ingestion_fixture_roundtrip_parquet(tmp_path) -> None:
    from pathlib import Path

    fixture = Path(__file__).with_name("fixtures").joinpath("events_roundtrip.jsonl")
    tbl = ingest_jsonl_to_canonical_table(fixture)
    validate_canonical_events_table(tbl)

    out_pq = tmp_path / "events.parquet"
    write_canonical_events_parquet(tbl, out_pq)

    tbl2 = read_canonical_events_parquet(out_pq)
    validate_canonical_events_table(tbl2)
    assert tbl2.equals(tbl)


def test_ingest_cell_id_null_handling(tmp_path) -> None:
    p = tmp_path / "events.jsonl"
    rows = [
        {"t": "2020-01-02", "lat": 51.5, "lon": -0.1, "cell_id": 7},
        {"t": "2020-01-03", "lat": 52.0, "lon": 0.1, "cell_id": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    tbl = ingest_jsonl_to_canonical_table(p)
    assert "cell_id" in tbl.column_names
    cell = tbl["cell_id"].to_pylist()
    assert cell == [7, None]


def test_validate_canonical_events_table_rejects_bad_schema() -> None:
    bad = pa.table({"t": [1], "lat": [0.0], "lon": [0.0], "value": [1]})
    # t must be date32, not int64.
    try:
        validate_canonical_events_table(bad)
    except ValueError as e:
        assert "t" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
