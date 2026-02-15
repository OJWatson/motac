"""Sparse neighbour convolution ops (JAX-jit ready, with NumPy fallback).

This module provides small, self-contained primitives used by the parametric
road-constrained Hawkes model:

- temporal convolution of per-cell count histories (lag kernel)
- sparse CSR matvec for neighbour aggregation

The project targets a JAX implementation for differentiable inference, but we
keep a pure-NumPy/SciPy-compatible path so the core package remains lightweight
and tests can run without JAX installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

try:  # optional
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


class ArrayLike(Protocol):
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CSR:
    """Minimal CSR container.

    The arrays mirror SciPy's CSR representation:
    - data: nonzeros
    - indices: column indices
    - indptr: index pointers (len = n_rows + 1)
    """

    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]


def csr_from_scipy(csr_matrix) -> CSR:
    """Convert a SciPy CSR matrix to a lightweight CSR container."""

    import scipy.sparse as sp

    if not sp.isspmatrix_csr(csr_matrix):
        csr_matrix = csr_matrix.tocsr()

    return CSR(
        data=np.asarray(csr_matrix.data),
        indices=np.asarray(csr_matrix.indices, dtype=np.int32),
        indptr=np.asarray(csr_matrix.indptr, dtype=np.int32),
        shape=(int(csr_matrix.shape[0]), int(csr_matrix.shape[1])),
    )


def convolved_history_last_numpy(*, y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Temporal convolution for the last time step (NumPy).

    For a history ``y[:, :T]``, returns

        h(T) = sum_{l=1..L} kernel[l-1] * y[:, T-l].

    Parameters
    ----------
    y:
        Array of shape (n_cells, T).
    kernel:
        Array of shape (L,). ``kernel[0]`` corresponds to lag 1.
    """

    if y.ndim != 2:
        raise ValueError("y must be 2D (n_cells, T)")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be 1D and non-empty")

    n_cells, t = y.shape
    effective = min(int(kernel.size), int(t))
    if effective == 0:
        return np.zeros((n_cells,), dtype=float)

    window = np.asarray(y[:, t - effective : t], dtype=float)
    k = np.asarray(kernel[:effective], dtype=float)
    return window[:, ::-1] @ k


def convolved_history_last_jax(*, y, kernel):
    """Temporal convolution for the last time step (JAX)."""

    if not _HAS_JAX:  # pragma: no cover
        raise RuntimeError("JAX is not available")

    if y.ndim != 2:
        raise ValueError("y must be 2D (n_cells, T)")
    if kernel.ndim != 1 or kernel.size == 0:
        raise ValueError("kernel must be 1D and non-empty")

    n_cells, t = y.shape
    effective = min(int(kernel.size), int(t))
    if effective == 0:
        return jnp.zeros((n_cells,), dtype=jnp.result_type(y, kernel, jnp.float32))

    window = y[:, t - effective : t]
    k = kernel[:effective]
    return jnp.flip(window, axis=1) @ k


def convolved_history_last(*, y: Any, kernel: Any) -> Any:
    """Dispatching temporal convolution.

    - If inputs are JAX arrays and JAX is available: use JAX.
    - Otherwise: use NumPy.
    """

    if _HAS_JAX:
        try:
            if isinstance(y, jnp.ndarray) or isinstance(kernel, jnp.ndarray):
                return convolved_history_last_jax(y=y, kernel=kernel)
        except Exception:  # pragma: no cover
            pass

    return convolved_history_last_numpy(y=np.asarray(y), kernel=np.asarray(kernel))


def csr_matvec_numpy(*, csr: CSR, x: np.ndarray) -> np.ndarray:
    """CSR matvec using NumPy.

    Intended for small/medium matrices in tests and CPU inference.
    """

    if x.ndim != 1:
        raise ValueError("x must be 1D")

    n_rows, n_cols = csr.shape
    if x.shape != (n_cols,):
        raise ValueError(f"x must have shape ({n_cols},)")

    y = np.zeros((n_rows,), dtype=np.result_type(csr.data, x, float))
    data = np.asarray(csr.data)
    indices = np.asarray(csr.indices, dtype=np.int32)
    indptr = np.asarray(csr.indptr, dtype=np.int32)

    for i in range(n_rows):
        lo = int(indptr[i])
        hi = int(indptr[i + 1])
        if lo == hi:
            continue
        y[i] = np.sum(data[lo:hi] * x[indices[lo:hi]])

    return y


def csr_matvec_jax(*, csr: CSR, x):
    """CSR matvec using JAX.

    Uses ``jax.experimental.sparse.BCSR`` when available.
    """

    if not _HAS_JAX:  # pragma: no cover
        raise RuntimeError("JAX is not available")

    if x.ndim != 1:
        raise ValueError("x must be 1D")

    n_rows, n_cols = csr.shape
    if tuple(x.shape) != (n_cols,):
        raise ValueError(f"x must have shape ({n_cols},)")

    # Prefer JAX sparse if present.
    try:
        from jax.experimental.sparse import BCSR

        data = jnp.asarray(csr.data)
        indices = jnp.asarray(csr.indices, dtype=jnp.int32)
        indptr = jnp.asarray(csr.indptr, dtype=jnp.int32)
        A = BCSR((data, indices, indptr), shape=csr.shape)
        return A @ x
    except Exception:
        # Fallback: explicit segment sum with row ids.
        data = jnp.asarray(csr.data)
        indices = jnp.asarray(csr.indices, dtype=jnp.int32)
        indptr = jnp.asarray(csr.indptr, dtype=jnp.int32)

        # Build row index per nonzero.
        # Note: this uses dynamic shapes, but is jit-friendly if csr is static.
        row_ids = jnp.repeat(jnp.arange(n_rows, dtype=jnp.int32), jnp.diff(indptr))
        contrib = data * x[indices]
        return jax.ops.segment_sum(contrib, row_ids, n_rows)


def csr_matvec(*, csr: CSR, x: Any) -> Any:
    """Dispatching CSR matvec.

    - If ``x`` is a JAX array and JAX is available: use JAX.
    - Otherwise: use NumPy.
    """

    if _HAS_JAX:
        try:
            import jax.numpy as _jnp

            if isinstance(x, _jnp.ndarray):
                return csr_matvec_jax(csr=csr, x=x)
        except Exception:  # pragma: no cover
            pass

    return csr_matvec_numpy(csr=csr, x=np.asarray(x))
