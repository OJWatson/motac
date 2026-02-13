from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motac.spatial.crs import LonLatToXY
from motac.substrate.types import Grid


@dataclass(frozen=True, slots=True)
class LonLatBounds:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def build_regular_grid(bounds: LonLatBounds, cell_size_m: float) -> Grid:
    if cell_size_m <= 0:
        raise ValueError('cell_size_m must be > 0')
    lon0 = 0.5 * (bounds.lon_min + bounds.lon_max)
    lat0 = 0.5 * (bounds.lat_min + bounds.lat_max)
    tf = LonLatToXY.for_lonlat(lon0, lat0)
    x0, y0 = tf.to_xy.transform(bounds.lon_min, bounds.lat_min)
    x1, y1 = tf.to_xy.transform(bounds.lon_max, bounds.lat_max)
    xmin, xmax = (min(x0, x1), max(x0, x1))
    ymin, ymax = (min(y0, y1), max(y0, y1))
    xs = np.arange(xmin + 0.5 * cell_size_m, xmax, cell_size_m)
    ys = np.arange(ymin + 0.5 * cell_size_m, ymax, cell_size_m)
    if xs.size == 0 or ys.size == 0:
        raise ValueError('bounds too small for given cell_size_m')
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = tf.to_ll.transform(xx.ravel(), yy.ravel())
    return Grid(
        lat=np.asarray(lat, dtype=float),
        lon=np.asarray(lon, dtype=float),
        cell_size_m=float(cell_size_m),
    )
