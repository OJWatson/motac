from datetime import date
from pathlib import Path

import pandas as pd

from motac.data import EventTable, GridSpec
from motac.datasets import (
    acled_to_counts,
    chicago_to_counts,
    fetch_acled_gaza,
    fetch_chicago_crimes,
)


def _sample_df():
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "event_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "latitude": [0.2, 0.8, 0.2],
            "longitude": [0.2, 0.8, 0.8],
            "primary_type": ["THEFT", "BATTERY", "THEFT"],
            "event_type": ["Battles", "Explosions/Remote violence", "Battles"],
        }
    )


def test_chicago_to_counts_shapes():
    events = EventTable(
        df=_sample_df(),
        time_col="date",
        lat_col="latitude",
        lon_col="longitude",
        mark_col="primary_type",
    )
    grid = GridSpec(shape=(2, 2), extent=(0.0, 1.0, 0.0, 1.0))
    counts = chicago_to_counts(events, grid)

    assert counts.y.shape[1] == 4
    assert counts.num_time == 2
    assert set(counts.marks) == {"BATTERY", "THEFT"}


def test_acled_to_counts_mark_mapping():
    events = EventTable(
        df=_sample_df(),
        time_col="event_date",
        lat_col="latitude",
        lon_col="longitude",
        mark_col="event_type",
    )
    grid = GridSpec(shape=(2, 2), extent=(0.0, 1.0, 0.0, 1.0))
    mapped = acled_to_counts(
        events,
        grid,
        mark_mapping={"Explosions/Remote violence": "Explosion"},
    )

    assert "Explosion" in mapped.marks


def test_fetchers_use_cache_without_network(tmp_path: Path):
    cdf = _sample_df()[["date", "latitude", "longitude",
                        "primary_type"]].copy()
    adf = _sample_df()[["event_date", "latitude",
                        "longitude", "event_type"]].copy()

    c_cache = tmp_path / "chicago_2024-01-01_2024-01-02.csv"
    a_cache = tmp_path / "acled_gaza_2024-01-01_2024-01-02.csv"
    cdf.to_csv(c_cache, index=False)
    adf.to_csv(a_cache, index=False)

    c_events = fetch_chicago_crimes(
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        cache_dir=tmp_path,
    )
    a_events = fetch_acled_gaza(
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        cache_dir=tmp_path,
    )

    assert len(c_events.df) == 3
    assert len(a_events.df) == 3
