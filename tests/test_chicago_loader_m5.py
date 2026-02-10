from __future__ import annotations

import numpy as np

from motac.chicago import load_y_obs_matrix


def test_load_y_obs_matrix_with_mobility(tmp_path) -> None:
    y_obs = np.array([[0, 1, 0], [2, 0, 1]], dtype=int)
    y_path = tmp_path / "y_obs.csv"
    np.savetxt(y_path, y_obs, fmt="%d", delimiter=",")

    mobility = np.array([[1.0, 0.2], [0.3, 1.0]], dtype=float)
    m_path = tmp_path / "mobility.npy"
    np.save(m_path, mobility)

    out = load_y_obs_matrix(path=y_path, mobility_path=m_path)

    assert out.y_obs.shape == (2, 3)
    assert np.array_equal(out.y_obs, y_obs)

    assert out.world.mobility.shape == (2, 2)
    assert np.allclose(out.world.mobility, mobility)

    assert out.meta["n_locations"] == 2
    assert out.meta["n_steps"] == 3
    assert out.meta["time_unit"] == "day"


def test_load_y_obs_matrix_defaults_identity_mobility(tmp_path) -> None:
    y_obs = np.array([[0, 0], [1, 2], [0, 1]], dtype=int)
    y_path = tmp_path / "y_obs.csv"
    np.savetxt(y_path, y_obs, fmt="%d", delimiter=",")

    out = load_y_obs_matrix(path=y_path)
    assert out.world.mobility.shape == (3, 3)
    assert np.allclose(out.world.mobility, np.eye(3))
