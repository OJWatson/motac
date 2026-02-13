from motac.substrate.builder import build_grid_from_lonlat_bounds


def test_build_grid_from_bounds_smoke():
    g = build_grid_from_lonlat_bounds(
        lon_min=0.0,
        lon_max=0.02,
        lat_min=51.49,
        lat_max=51.51,
        cell_size_m=500.0,
    )
    assert g.lat.size == g.lon.size
    assert g.cell_size_m == 500.0
