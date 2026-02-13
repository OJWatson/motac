from __future__ import annotations

import numpy as np
import pytest

from motac.schema import EventRecord, EventTable


def test_event_table_validates_and_roundtrips() -> None:
    recs = [
        EventRecord(
            event_id="e1",
            t="2020-01-01",
            lat=1.0,
            lon=2.0,
            cell_id=5,
            mark="a",
            value=1,
        ),
        EventRecord(
            event_id="e2",
            t="2020-01-02",
            lat=1.0,
            lon=2.0,
            cell_id=None,
            mark="a",
            value=3,
        ),
        EventRecord(
            event_id=None,
            t="2020-01-02",
            lat=9.0,
            lon=8.0,
            cell_id=0,
            mark=None,
            value=0,
        ),
    ]

    tab = EventTable.from_records(recs)
    tab.validate()

    assert tab.t.shape == (3,)
    assert tab.lat.shape == (3,)
    assert tab.lon.shape == (3,)
    assert tab.value.shape == (3,)

    assert tab.t.dtype.kind == "M"  # datetime64
    assert np.all(tab.value >= 0)

    recs2 = tab.to_records()
    assert recs2[0].event_id == "e1"
    assert recs2[0].cell_id == 5
    assert recs2[1].cell_id is None  # preserved via -1 sentinel


def test_event_table_rejects_bad_lat_lon() -> None:
    recs = [EventRecord(t="2020-01-01", lat=999.0, lon=2.0, value=1)]
    tab = EventTable.from_records(recs)
    with pytest.raises(ValueError, match="lat out of range"):
        tab.validate()


def test_event_table_rejects_negative_value() -> None:
    recs = [EventRecord(t="2020-01-01", lat=1.0, lon=2.0, value=-1)]
    tab = EventTable.from_records(recs)
    with pytest.raises(ValueError, match="value must be non-negative"):
        tab.validate()
