from __future__ import annotations

import numpy as np
import pytest

from motac.model.marked_hawkes import encode_categorical_marks_onehot


def test_encode_categorical_marks_onehot_shape_dtype_and_values() -> None:
    y_obs = np.zeros((2, 3), dtype=int)
    marks = np.array([[0, 1, 0], [1, 0, 1]], dtype=int)

    out = encode_categorical_marks_onehot(marks, y_obs=y_obs, n_marks=2)

    assert out.shape == (2, 3, 2)
    assert out.dtype == np.float32

    # spot-check values
    assert np.array_equal(out[0, 0], np.array([1.0, 0.0], dtype=np.float32))
    assert np.array_equal(out[0, 1], np.array([0.0, 1.0], dtype=np.float32))
    assert np.array_equal(out[1, 2], np.array([0.0, 1.0], dtype=np.float32))


def test_encode_categorical_marks_onehot_rejects_invalid_n_marks() -> None:
    y_obs = np.zeros((1, 1), dtype=int)
    marks = np.zeros_like(y_obs)

    with pytest.raises(ValueError, match="n_marks must be positive"):
        _ = encode_categorical_marks_onehot(marks, y_obs=y_obs, n_marks=0)
