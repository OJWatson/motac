"""Neural / learned kernel scaffolding.

This module intentionally starts small: it establishes a stable import path and a
minimal kernel interface for future nonparametric / neural Hawkes variants.

The goal (M14) is *not* to ship a full neural model; it is to make downstream
code able to depend on a simple, documented call signature and shape contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class KernelFn(Protocol):
    """A minimal kernel function interface.

    Contract (v1 scaffold): kernels map a nonnegative distance / travel-time
    tensor to a same-shaped, nonnegative weight tensor.

    Implementations should be pure and deterministic.
    """

    def __call__(self, d: np.ndarray) -> np.ndarray: ...


def validate_kernel_fn(kernel: KernelFn, *, name: str = "kernel") -> None:
    """Validate that a kernel satisfies the minimal (v1) shape/value contract.

    This is intentionally small and opinionated: it exists to catch accidental
    contract drift early (e.g. returning negative weights or wrong shapes).

    Parameters
    ----------
    kernel:
        Callable to validate.
    name:
        Used in raised error messages.
    """

    d = np.asarray([[0.0, 1.0, 2.5], [0.0, 0.5, 3.0]], dtype=np.float64)
    w = kernel(d)

    if not isinstance(w, np.ndarray):
        raise TypeError(f"{name} must return a numpy.ndarray, got {type(w).__name__}")
    if w.shape != d.shape:
        raise ValueError(f"{name} must return same shape as input: {w.shape} != {d.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must return finite weights")
    if np.any(w < 0):
        raise ValueError(f"{name} must return nonnegative weights")


@dataclass(frozen=True)
class ExpDecayKernel:
    """A tiny, deterministic kernel implementation for unit tests.

    Computes ``w = exp(-d / lengthscale)`` elementwise.

    Parameters
    ----------
    lengthscale:
        Positive scale parameter controlling the rate of decay.
    """

    lengthscale: float = 1.0

    def __call__(self, d: np.ndarray) -> np.ndarray:
        if self.lengthscale <= 0:
            raise ValueError("lengthscale must be positive")

        x = np.asarray(d, dtype=np.float64)
        if np.any(x < 0):
            raise ValueError("d must be nonnegative")

        out = np.exp(-x / float(self.lengthscale))
        return out
