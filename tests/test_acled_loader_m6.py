from __future__ import annotations

import numpy as np

from motac.acled import load_acled_events_csv


def test_load_acled_events_csv_aggregates_events_and_fatalities(tmp_path) -> None:
    text = (
        "event_date,lat,lon,fatalities\n"
        "2020-01-01,1.0,2.0,0\n"
        "2020-01-01,1.0,2.0,3\n"
        "2020-01-02,1.0,2.0,1\n"
        "2020-01-02,9.0,8.0,2\n"
    )
    path = tmp_path / "acled.csv"
    path.write_text(text)

    mobility = np.array([[1.0, 0.1], [0.2, 1.0]], dtype=float)
    m_path = tmp_path / "mobility.npy"
    np.save(m_path, mobility)

    out_events = load_acled_events_csv(path=path, mobility_path=m_path, value="events")
    assert out_events.y_obs.shape == (2, 2)
    # location (2,1): 2 events day1, 1 event day2
    # location (8,9): 0 events day1, 1 event day2
    assert np.array_equal(out_events.y_obs, np.array([[2, 1], [0, 1]]))
    assert out_events.world.mobility.shape == (2, 2)
    assert np.allclose(out_events.world.mobility, mobility)
    assert out_events.meta["dates"] == ["2020-01-01", "2020-01-02"]

    out_fat = load_acled_events_csv(path=path, mobility_path=m_path, value="fatalities")
    assert np.array_equal(out_fat.y_obs, np.array([[3, 1], [0, 2]]))
