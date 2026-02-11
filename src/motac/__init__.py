"""Road-constrained spatio-temporal Hawkes processes (JAX).

This package implements a research-grade, reproducible pipeline for
road-network-constrained Hawkes models, simulators, and benchmarks.
"""

from __future__ import annotations

from ._version import __version__
from .eval import EvalConfig, evaluate_synthetic
from .loaders import AcledData, ChicagoData, load_acled_events_csv, load_y_obs_matrix
from .model import (
    RoadHawkesDataset,
    fit_road_hawkes_mle,
    forecast_intensity_horizon,
    mean_negative_log_likelihood,
    predict_intensity_in_sample,
    predict_intensity_next_step,
    predict_intensity_one_step_road,
    road_loglik,
)
from .neural import NeuralIntensityBaseline

__all__ = [
    "__version__",
    "EvalConfig",
    "evaluate_synthetic",
    "ChicagoData",
    "load_y_obs_matrix",
    "AcledData",
    "load_acled_events_csv",
    "RoadHawkesDataset",
    "predict_intensity_one_step_road",
    "predict_intensity_in_sample",
    "predict_intensity_next_step",
    "forecast_intensity_horizon",
    "mean_negative_log_likelihood",
    "road_loglik",
    "fit_road_hawkes_mle",
    "NeuralIntensityBaseline",
]
