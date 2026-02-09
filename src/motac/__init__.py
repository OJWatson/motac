"""Road-constrained spatio-temporal Hawkes processes (JAX).

This package implements a research-grade, reproducible pipeline for
road-network-constrained Hawkes models, simulators, and benchmarks.
"""

from __future__ import annotations

from ._version import __version__
from .eval import EvalConfig, evaluate_synthetic

__all__ = ["__version__", "EvalConfig", "evaluate_synthetic"]
