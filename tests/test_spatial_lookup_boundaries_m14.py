from __future__ import annotations

import numpy as np

from motac.spatial.lookup import GridCellLookup
from motac.substrate.builder import build_grid_from_lonlat_bounds


def test_lonlat_to_cell_id_boundary_convention_edges_inclusive_exclusive():
    """Regression test for boundary convention.

    Expected behaviour for the regular grid is:
    - left/bottom edges are inclusive
    - right/top edges are exclusive

    This matches np.arange(xmin+0.5*cs, xmax, cs) in grid construction and the
    floor-based indexing in GridCellLookup.
    """

    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.2,
        lon_max=0.2,
        lat_min=51.4,
        lat_max=51.6,
        cell_size_m=5_000.0,
    )
    lu = GridCellLookup.from_grid(grid)

    x0 = float(lu.x0_edge)
    y0 = float(lu.y0_edge)
    x1 = x0 + lu.nx * lu.cell_size_m
    y1 = y0 + lu.ny * lu.cell_size_m

    # Lower-left corner sits exactly on the inclusive edges.
    lon_ll, lat_ll = lu.tf.to_ll.transform(x0, y0)
    assert lu.lonlat_to_cell_id(lon=lon_ll, lat=lat_ll) == 0

    # Upper-right corner is on exclusive edges => outside.
    lon_ur, lat_ur = lu.tf.to_ll.transform(x1, y1)
    assert lu.lonlat_to_cell_id(lon=lon_ur, lat=lat_ur) == -1

    # Just inside upper-right should map to last cell.
    eps = 1e-6 * lu.cell_size_m
    lon_in, lat_in = lu.tf.to_ll.transform(x1 - eps, y1 - eps)
    assert lu.lonlat_to_cell_id(lon=lon_in, lat=lat_in) == (lu.nx * lu.ny - 1)

    # Right edge exclusive.
    lon_r, lat_r = lu.tf.to_ll.transform(x1, 0.5 * (y0 + y1))
    assert lu.lonlat_to_cell_id(lon=lon_r, lat=lat_r) == -1

    # Top edge exclusive.
    lon_t, lat_t = lu.tf.to_ll.transform(0.5 * (x0 + x1), y1)
    assert lu.lonlat_to_cell_id(lon=lon_t, lat=lat_t) == -1


def test_lonlat_to_cell_id_matches_naive_indexing_for_random_points():
    """Property-style test: lookup matches naive floor indexing in projected space."""

    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.15,
        lon_max=0.15,
        lat_min=51.45,
        lat_max=51.65,
        cell_size_m=7_500.0,
    )
    lu = GridCellLookup.from_grid(grid)

    x0 = float(lu.x0_edge)
    y0 = float(lu.y0_edge)
    x1 = x0 + lu.nx * lu.cell_size_m
    y1 = y0 + lu.ny * lu.cell_size_m

    rng = np.random.default_rng(0)

    # Draw random points inside the grid bounds (exclusive of right/top edges).
    n = 2_000
    x = rng.uniform(low=x0, high=np.nextafter(x1, x0), size=n)
    y = rng.uniform(low=y0, high=np.nextafter(y1, y0), size=n)
    lon, lat = lu.tf.to_ll.transform(x, y)

    out = lu.lonlat_to_cell_id(lon=lon, lat=lat)
    assert out.shape == (n,)
    assert np.all(out >= 0)
    assert np.all(out < lu.nx * lu.ny)

    ix = np.floor((x - x0) / lu.cell_size_m).astype(int)
    iy = np.floor((y - y0) / lu.cell_size_m).astype(int)
    expected = iy * lu.nx + ix

    assert np.array_equal(out, expected)


def test_lonlat_to_cell_id_returns_minus_one_for_outside_random_points():
    """Property-style test: outside points return -1 (vectorised)."""

    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.1,
        lon_max=0.1,
        lat_min=51.4,
        lat_max=51.6,
        cell_size_m=10_000.0,
    )
    lu = GridCellLookup.from_grid(grid)

    x0 = float(lu.x0_edge)
    y0 = float(lu.y0_edge)
    x1 = x0 + lu.nx * lu.cell_size_m
    y1 = y0 + lu.ny * lu.cell_size_m

    rng = np.random.default_rng(1)

    n = 500
    # Points far to the right of the grid.
    x = rng.uniform(low=x1 + lu.cell_size_m, high=x1 + 10 * lu.cell_size_m, size=n)
    y = rng.uniform(low=y0, high=np.nextafter(y1, y0), size=n)
    lon, lat = lu.tf.to_ll.transform(x, y)

    out = lu.lonlat_to_cell_id(lon=lon, lat=lat)
    assert np.array_equal(out, -np.ones((n,), dtype=int))
