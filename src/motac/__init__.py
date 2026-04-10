"""Public package exports for motac."""

from .acled_variants import (
    AcledSpatialVariant,
    aggregate_variant_counts_to_map,
    build_acled_spatial_variant,
    build_standard_acled_variants,
    flatten_variant_counts,
    summarise_variant_zero_mass,
    variant_node_values_to_map,
)
from .connectivity import (
    ConvStencil,
    EdgeList,
    apply_conv_stencil,
    apply_edgelist,
    apply_spatial_backend,
    make_grid_edgelist,
    make_masked_grid_edgelist,
    make_grid_stencil,
    subset_edgelist,
)
from .bench import (
    BenchmarkResult,
    benchmark_backend,
    benchmarks_to_frame,
    compare_backends,
    save_benchmarks_csv,
)
from .data import CountsTensor, EventTable, GridSpec, TrainTestSplit
from .datasets import acled_to_counts, chicago_to_counts, fetch_acled_gaza, fetch_chicago_crimes
from .eval import (
    BacktestResult,
    aggregate_daily_totals,
    aggregate_mark_totals,
    coverage,
    hotspot_recall,
    rolling_backtest,
    score_crps_counts,
    score_log_likelihood_nb2,
    score_predictive_log_prob_samples,
)
from .forecast import ForecastResult
from .infer import FitConfig, FitResult
from .model import HawkesModelSpec, MobilityHawkesModel
from .simulate import HawkesSimNoise, HawkesSimParams, counts_to_events, simulate_counts

__all__ = [
    "BacktestResult",
    "BenchmarkResult",
    "AcledSpatialVariant",
    "ConvStencil",
    "CountsTensor",
    "EdgeList",
    "EventTable",
    "build_acled_spatial_variant",
    "build_standard_acled_variants",
    "flatten_variant_counts",
    "variant_node_values_to_map",
    "aggregate_variant_counts_to_map",
    "summarise_variant_zero_mass",
    "aggregate_daily_totals",
    "aggregate_mark_totals",
    "coverage",
    "hotspot_recall",
    "score_predictive_log_prob_samples",
    "score_crps_counts",
    "score_log_likelihood_nb2",
    "fetch_chicago_crimes",
    "fetch_acled_gaza",
    "chicago_to_counts",
    "acled_to_counts",
    "benchmark_backend",
    "benchmarks_to_frame",
    "compare_backends",
    "save_benchmarks_csv",
    "FitConfig",
    "FitResult",
    "ForecastResult",
    "GridSpec",
    "HawkesModelSpec",
    "HawkesSimNoise",
    "HawkesSimParams",
    "MobilityHawkesModel",
    "TrainTestSplit",
    "counts_to_events",
    "apply_conv_stencil",
    "apply_edgelist",
    "apply_spatial_backend",
    "subset_edgelist",
    "make_masked_grid_edgelist",
    "make_grid_stencil",
    "make_grid_edgelist",
    "rolling_backtest",
    "simulate_counts",
]
