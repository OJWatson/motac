from __future__ import annotations

import numpy as np

from motac.sim import summarize_predictive_paths


def test_summarize_predictive_paths_toy() -> None:
    # Two paths over a (locations, horizon) array.
    paths = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )  # shape (2, 2, 2)

    out = summarize_predictive_paths(paths=paths, q=(0.0, 0.5, 1.0))

    assert out["mean"].shape == (2, 2)
    assert np.allclose(out["mean"], np.array([[1.0, 2.0], [3.0, 4.0]]))

    qu = out["quantiles"]
    assert qu.shape == (3, 2, 2)
    assert np.allclose(qu[0], paths.min(axis=0))
    assert np.allclose(qu[2], paths.max(axis=0))
