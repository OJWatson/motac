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
    fit_hawkes_mle_alpha_mu_observed_poisson_approx,
    fit_observation_params_exact,
)
from .hawkes import (
    HawkesDiscreteParams,
    discrete_exponential_kernel,
    predict_hawkes_intensity_multi_step,
    predict_hawkes_intensity_one_step,
    sample_hawkes_observed_predictive_paths_poisson_approx,
    sample_hawkes_predictive_paths,
    simulate_hawkes_counts,
)
from .io import load_simulation_parquet, save_simulation_parquet
from .likelihood import (
    hawkes_intensity,
    hawkes_loglik_observed_exact,
    hawkes_loglik_poisson,
    hawkes_loglik_poisson_observed,
)
from .predictive import summarize_predictive_paths
from .world import World, generate_random_world

__all__ = [
    "World",
    "generate_random_world",
    "HawkesDiscreteParams",
    "discrete_exponential_kernel",
    "simulate_hawkes_counts",
    "predict_hawkes_intensity_one_step",
    "predict_hawkes_intensity_multi_step",
    "sample_hawkes_predictive_paths",
    "sample_hawkes_observed_predictive_paths_poisson_approx",
    "hawkes_intensity",
    "hawkes_loglik_poisson",
    "hawkes_loglik_poisson_observed",
    "hawkes_loglik_observed_exact",
    "summarize_predictive_paths",
    "fit_hawkes_alpha_mu",
    "fit_hawkes_mle_alpha_mu",
    "fit_hawkes_mle_alpha_mu_beta",
    "fit_hawkes_mle_alpha_mu_observed_poisson_approx",
    "fit_observation_params_exact",
    "save_simulation_parquet",
    "load_simulation_parquet",
]
