from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motac.substrate.features import min_travel_time_feature_matrix, min_travel_time_to_mask


def test_min_travel_time_to_mask_toy() -> None:
    # 3 cells, directed-ish travel times.
    mat = sp.csr_matrix(
        np.array(
            [
                [0.0, 5.0, 9.0],
                [5.0, 0.0, 2.0],
                [9.0, 2.0, 0.0],
            ]
        )
    )
    mask = np.array([False, True, False])

    out = min_travel_time_to_mask(travel_time_s=mat, mask=mask, default=99.0)
    assert out.shape == (3,)
    # min time to cell 1 from each i.
    assert np.allclose(out, np.array([5.0, 0.0, 2.0]))


def test_min_travel_time_to_mask_default_when_none() -> None:
    mat = sp.csr_matrix(np.eye(2))
    out = min_travel_time_to_mask(travel_time_s=mat, mask=np.array([False, False]), default=7.0)
    assert np.allclose(out, np.array([7.0, 7.0]))


def test_min_travel_time_feature_matrix_multiple_masks() -> None:
    mat = sp.csr_matrix(
        np.array(
            [
                [0.0, 5.0, 9.0],
                [5.0, 0.0, 2.0],
                [9.0, 2.0, 0.0],
            ]
        )
    )

    masks = {
        "poi": np.array([False, True, False]),
        "poi_school": np.array([False, False, True]),
    }

    x, names = min_travel_time_feature_matrix(
        travel_time_s=mat, masks=masks, default=99.0, suffix="min_travel_time_s"
    )

    assert names == ["poi_min_travel_time_s", "poi_school_min_travel_time_s"]
    assert x.shape == (3, 2)
    assert np.allclose(x[:, 0], np.array([5.0, 0.0, 2.0]))
    assert np.allclose(x[:, 1], np.array([9.0, 2.0, 0.0]))
