"""Simulation utilities.

Milestone M2 introduces a lightweight discrete-time simulator for a
network-coupled Hawkes-like count process.
"""

from __future__ import annotations

from .fit import fit_hawkes_alpha_mu, fit_hawkes_mle_alpha_mu
from .hawkes import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    simulate_hawkes_counts,
)
from .io import load_simulation_parquet, save_simulation_parquet
from .likelihood import hawkes_intensity, hawkes_loglik_poisson
from .world import World, generate_random_world

__all__ = [
    "World",
    "generate_random_world",
    "HawkesDiscreteParams",
    "discrete_exponential_kernel",
    "simulate_hawkes_counts",
    "hawkes_intensity",
    "hawkes_loglik_poisson",
    "fit_hawkes_alpha_mu",
    "fit_hawkes_mle_alpha_mu",
    "save_simulation_parquet",
    "load_simulation_parquet",
]
