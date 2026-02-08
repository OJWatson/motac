"""Simulation utilities.

Milestones
----------
M2: Discrete-time simulator for a network-coupled Hawkes-like count process.
M4: Parametric prediction API for one-step and multi-step intensity forecasts
    given a history of counts.
"""

from __future__ import annotations

from .fit import (
    fit_hawkes_alpha_mu,
    fit_hawkes_mle_alpha_mu,
    fit_hawkes_mle_alpha_mu_beta,
)
from .hawkes import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    predict_hawkes_intensity_multi_step,
    predict_hawkes_intensity_one_step,
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
    "predict_hawkes_intensity_one_step",
    "predict_hawkes_intensity_multi_step",
    "hawkes_intensity",
    "hawkes_loglik_poisson",
    "fit_hawkes_alpha_mu",
    "fit_hawkes_mle_alpha_mu",
    "fit_hawkes_mle_alpha_mu_beta",
    "save_simulation_parquet",
    "load_simulation_parquet",
]
