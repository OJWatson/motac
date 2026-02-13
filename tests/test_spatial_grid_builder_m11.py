import numpy as np

from motac.spatial.crs import LonLatToXY
from motac.spatial.grid_builder import LonLatBounds, build_regular_grid


def test_lonlat_to_xy_roundtrip():
    tf = LonLatToXY.for_lonlat(lon0=0.1, lat0=51.5)
    lon = np.array([0.1, 0.2])
    lat = np.array([51.5, 51.6])
    x, y = tf.to_xy.transform(lon, lat)
    lon2, lat2 = tf.to_ll.transform(x, y)
    assert np.allclose(lon2, lon, atol=1e-6)
    assert np.allclose(lat2, lat, atol=1e-6)


def test_build_regular_grid_smoke():
    b = LonLatBounds(lon_min=0.0, lon_max=0.02, lat_min=51.49, lat_max=51.51)
    g = build_regular_grid(b, cell_size_m=500.0)
    assert g.lat.ndim == 1
    assert g.lon.ndim == 1
    assert g.lat.size == g.lon.size
    assert g.cell_size_m == 500.0
