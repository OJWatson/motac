from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from motac.sim import SubstrateHawkesParams, simulate_substrate_hawkes_counts
from motac.substrate.types import Grid, NeighbourSets, Substrate


def test_substrate_hawkes_sim_matches_golden_fixture() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "m4_0"
    golden = np.load(fixtures / "substrate_hawkes_golden.npz")

    # Small, deterministic substrate.
    travel_time_s = sp.csr_matrix(
        np.array(
            [
                [0.0, 10.0, 30.0],
                [10.0, 0.0, 15.0],
                [30.0, 15.0, 0.0],
            ]
        )
    )

    grid = Grid(
        lat=np.array([0.0, 0.0, 0.0]),
        lon=np.array([0.0, 0.0, 0.0]),
        cell_size_m=1000.0,
    )
    substrate = Substrate(
        grid=grid,
        neighbours=NeighbourSets(travel_time_s=travel_time_s),
        poi=None,
        graphml_path=None,
    )

    params = SubstrateHawkesParams(
        mu=np.array([0.08, 0.10, 0.06]),
        alpha=0.7,
        beta=0.02,
        kernel=np.array([1.0, 0.5, 0.25]),
    )

    out = simulate_substrate_hawkes_counts(substrate=substrate, params=params, n_steps=25, seed=123)

    assert out["y_true"].shape == (3, 25)
    assert out["intensity"].shape == (3, 25)

    assert np.array_equal(out["y_true"], golden["y_true"])
    assert np.allclose(out["intensity"], golden["intensity"], rtol=0.0, atol=0.0)
