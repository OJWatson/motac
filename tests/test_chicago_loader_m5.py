from __future__ import annotations

from pathlib import Path

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
    assert out.meta["mobility_source"] == "identity"
    assert out.world.mobility.shape == (3, 3)
    assert np.allclose(out.world.mobility, np.eye(3))


def test_load_y_obs_matrix_preserves_row_order(tmp_path) -> None:
    # Contract: row i corresponds to location index i; loader must not sort.
    y_obs = np.array([[1, 0], [9, 0], [5, 0]], dtype=int)
    perm = [2, 0, 1]
    y_perm = y_obs[perm]

    y_path = tmp_path / "y_obs_perm.csv"
    np.savetxt(y_path, y_perm, fmt="%d", delimiter=",")

    out = load_y_obs_matrix(path=y_path)
    assert np.array_equal(out.y_obs, y_perm)


def test_load_y_obs_matrix_from_repo_fixture() -> None:
    y_fixture = Path(__file__).parent / "fixtures" / "chicago" / "y_obs_small.csv"
    m_fixture = Path(__file__).parent / "fixtures" / "chicago" / "mobility_small.npy"

    out = load_y_obs_matrix(path=y_fixture, mobility_path=m_fixture)

    assert out.meta["schema"] == "placeholder-y_obs-matrix"
    assert out.meta["mobility_source"] == str(m_fixture)

    assert out.y_obs.shape == (2, 4)
    assert np.array_equal(out.y_obs, np.array([[0, 1, 0, 2], [3, 0, 0, 1]], dtype=int))

    assert out.world.mobility.shape == (2, 2)
    assert np.allclose(out.world.mobility, np.array([[1.0, 0.1], [0.2, 1.0]]))


def test_load_y_obs_matrix_from_raw_dir_autodetects_mobility(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    y_obs = np.array([[0, 1, 0], [2, 0, 1]], dtype=int)
    np.savetxt(raw_dir / "y_obs.csv", y_obs, fmt="%d", delimiter=",")

    mobility = np.array([[1.0, 0.2], [0.3, 1.0]], dtype=float)
    np.save(raw_dir / "mobility.npy", mobility)

    out = load_y_obs_matrix(path=raw_dir)

    assert np.array_equal(out.y_obs, y_obs)
    assert np.allclose(out.world.mobility, mobility)
    assert out.meta["raw_dir"] == str(raw_dir)


def test_load_y_obs_matrix_from_raw_dir_defaults_identity_when_missing(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    y_obs = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    np.savetxt(raw_dir / "y_obs.csv", y_obs, fmt="%d", delimiter=",")

    out = load_y_obs_matrix(path=raw_dir)

    assert out.meta["mobility_source"] == "identity"
    assert np.allclose(out.world.mobility, np.eye(3))
    assert out.meta["raw_dir"] == str(raw_dir)


def test_load_y_obs_matrix_rejects_mismatched_mobility_shape(tmp_path) -> None:
    y_obs = np.array([[0, 1, 0], [2, 0, 1]], dtype=int)
    y_path = tmp_path / "y_obs.csv"
    np.savetxt(y_path, y_obs, fmt="%d", delimiter=",")

    mobility = np.eye(3, dtype=float)
    m_path = tmp_path / "mobility_bad.npy"
    np.save(m_path, mobility)

    try:
        load_y_obs_matrix(path=y_path, mobility_path=m_path)
    except ValueError as e:
        assert "mobility must have shape" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatched mobility shape")
