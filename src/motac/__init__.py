"""Road-constrained spatio-temporal Hawkes processes (JAX).

This package implements a research-grade, reproducible pipeline for
road-network-constrained Hawkes models, simulators, and benchmarks.
"""

from __future__ import annotations

from ._version import __version__
from .acled import AcledData, load_acled_events_csv
from .chicago import ChicagoData, load_y_obs_matrix
from .eval import EvalConfig, evaluate_synthetic
from .model import (
    RoadHawkesDataset,
    fit_road_hawkes_mle,
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
    "road_loglik",
    "fit_road_hawkes_mle",
    "NeuralIntensityBaseline",
]
