from __future__ import annotations

import json

import pyarrow as pa

from motac.ingestion import ingest_jsonl_to_canonical_table


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
    assert tbl.column_names[:4] == ["t", "lat", "lon", "value"]
    assert tbl.schema.field("t").type == pa.date32()
