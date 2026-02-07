from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True, slots=True)
class Grid:
    """Regular grid substrate.

    Coordinates are stored as WGS84 centroids for downstream modelling.
    """

    # shape: (n_cells,)
    lat: np.ndarray
    lon: np.ndarray
    # cell size (metres) in the local projected CRS used to build grid
    cell_size_m: float


@dataclass(frozen=True, slots=True)
class POIFeatures:
    """POI features per grid cell."""

    # shape: (n_cells, n_features)
    x: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True, slots=True)
class NeighbourSets:
    """Sparse travel-time neighbourhoods between grid cells.

    matrix[i, j] = travel time in seconds from i to j (0 on diagonal).
    """

    travel_time_s: sp.csr_matrix


@dataclass(frozen=True, slots=True)
class Substrate:
    grid: Grid
    neighbours: NeighbourSets
    poi: POIFeatures | None
    graphml_path: str | None = None
