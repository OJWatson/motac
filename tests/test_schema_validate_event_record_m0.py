from __future__ import annotations

import pytest

from motac.schema import EventRecord, validate_event_record


def test_validate_event_record_rejects_bad_date() -> None:
    with pytest.raises(ValueError, match="t must be YYYY-MM-DD"):
        validate_event_record(EventRecord(t="not-a-date", lat=1.0, lon=2.0))


def test_validate_event_record_rejects_empty_mark() -> None:
    with pytest.raises(ValueError, match="mark must be"):
        validate_event_record(EventRecord(t="2020-01-01", lat=1.0, lon=2.0, mark=""))


def test_validate_event_record_rejects_bad_event_id() -> None:
    with pytest.raises(ValueError, match="event_id must be"):
        validate_event_record(EventRecord(event_id="", t="2020-01-01", lat=1.0, lon=2.0))


def test_validate_event_record_rejects_negative_cell_id() -> None:
    with pytest.raises(ValueError, match="cell_id must be"):
        validate_event_record(EventRecord(t="2020-01-01", lat=1.0, lon=2.0, cell_id=-3))
