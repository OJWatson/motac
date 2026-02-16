"""Inference utilities.

This package is a structure-first boundary for inference-related code:
likelihoods, constraints/parameterisations, and optimisers.

It also hosts small, composable primitives that are designed to be JAX-jittable
when JAX is installed, while keeping a lightweight NumPy fallback.
"""

from __future__ import annotations

from .likelihoods import negbin_loglik, negbin_logpmf, poisson_loglik, poisson_logpmf
from .sparse_neighbour_ops import CSR, convolved_history_last, csr_from_scipy, csr_matvec

__all__ = [
    "CSR",
    "convolved_history_last",
    "csr_from_scipy",
    "csr_matvec",
    "negbin_loglik",
    "negbin_logpmf",
    "poisson_loglik",
    "poisson_logpmf",
]
