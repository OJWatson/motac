from __future__ import annotations

import numpy as np


def test_neural_kernels_scaffold_import_path_stable() -> None:
    from motac.model import neural_kernels  # noqa: F401
    from motac.model.neural_kernels import ExpDecayKernel  # noqa: F401


def test_exp_decay_kernel_toy_contract() -> None:
    from motac.model.neural_kernels import ExpDecayKernel

    k = ExpDecayKernel(lengthscale=2.0)
    d = np.asarray([0.0, 2.0, 4.0])
    w = k(d)

    assert w.shape == d.shape
    assert np.all(w >= 0)
    np.testing.assert_allclose(w, np.exp(-d / 2.0))
