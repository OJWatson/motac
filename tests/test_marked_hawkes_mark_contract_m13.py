from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pytest

from motac.model.dataset import RoadHawkesDataset
from motac.model.marked_hawkes import (
    MarkedRoadHawkesDataset,
    validate_categorical_marks_matrix,
)
from motac.substrate import SubstrateBuilder, SubstrateConfig


def _write_tiny_graphml(path: Path) -> None:
    # three nodes roughly in a line in WGS84
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"

    G.add_node(0, x=-0.1000, y=51.5000)
    G.add_node(1, x=-0.0990, y=51.5000)
    G.add_node(2, x=-0.0980, y=51.5000)

    # 60s per hop
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 2, key=0, travel_time=60.0, length=100.0)
    # bidirectional
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    G.add_edge(2, 1, key=0, travel_time=60.0, length=100.0)

    ox.save_graphml(G, filepath=path)


def test_mark_contract_v1_shape_and_dtype_validation(tmp_path: Path) -> None:
    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=61.0,
        disable_pois=True,
        cache_dir=str(tmp_path / "cache"),
    )
    sub = SubstrateBuilder(cfg).build()

    y_obs = np.zeros((len(sub.grid.lat), 3), dtype=int)
    base = RoadHawkesDataset(substrate=sub, y_obs=y_obs)

    # shape mismatch
    with pytest.raises(ValueError, match="marks must match"):
        _ = MarkedRoadHawkesDataset(base=base, marks=np.zeros((base.n_cells, 2), dtype=int))

    # non-integer dtype
    with pytest.raises(ValueError, match="integer dtype"):
        _ = validate_categorical_marks_matrix(
            np.zeros_like(y_obs, dtype=float),
            y_obs=y_obs,
        )

    # negative values
    bad = np.zeros_like(y_obs, dtype=int)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        _ = validate_categorical_marks_matrix(bad, y_obs=y_obs)

    # optional bound
    ok = np.zeros_like(y_obs, dtype=int)
    _ = validate_categorical_marks_matrix(ok, y_obs=y_obs, n_marks=1)
    too_big = ok.copy()
    too_big[0, 1] = 3
    with pytest.raises(ValueError, match=r"\[0, n_marks\)"):
        _ = validate_categorical_marks_matrix(too_big, y_obs=y_obs, n_marks=2)
