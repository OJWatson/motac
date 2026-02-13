from __future__ import annotations

import numpy as np
import pytest


def test_validate_kernel_fn_accepts_exp_decay_kernel() -> None:
    from motac.model.neural_kernels import ExpDecayKernel, validate_kernel_fn

    validate_kernel_fn(ExpDecayKernel(lengthscale=1.5))


def test_validate_kernel_fn_rejects_negative_weights() -> None:
    from motac.model.neural_kernels import validate_kernel_fn

    def bad_kernel(d: np.ndarray) -> np.ndarray:
        return -np.ones_like(d, dtype=np.float64)

    with pytest.raises(ValueError, match="nonnegative"):
        validate_kernel_fn(bad_kernel, name="bad_kernel")


def test_validate_kernel_fn_rejects_wrong_shape() -> None:
    from motac.model.neural_kernels import validate_kernel_fn

    def bad_kernel(d: np.ndarray) -> np.ndarray:
        return np.ones((d.shape[0],), dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        validate_kernel_fn(bad_kernel, name="bad_kernel")
