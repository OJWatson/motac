"""Backtesting and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data import CountsTensor, TrainTestSplit
from .forecast import ForecastResult
from .infer import FitConfig
from .model import MobilityHawkesModel


@dataclass(frozen=True)
class BacktestResult:
    """Container holding rolling-origin forecast artifacts and summary metrics."""

    splits: list[TrainTestSplit]
    forecasts: list[ForecastResult]
    metrics: pd.DataFrame
    meta: dict[str, Any]


def score_predictive_log_prob_samples(y_true: jnp.ndarray, y_samples: jnp.ndarray) -> float:
    """Sample-based predictive log-score with Laplace smoothing.

    For each target count value at each forecast site, estimate probability mass from
    predictive samples:
    p(y) ~= (matches + alpha) / (S + alpha * K)
    where K is the number of unique predictive outcomes at that site.
    """
    y_true_np = np.asarray(y_true)
    y_samp_np = np.asarray(y_samples)
    s = y_samp_np.shape[0]
    if s <= 0:
        raise ValueError("y_samples must contain at least one sample")

    alpha = 1.0

    # [S, ...] compare predictive samples to observed targets.
    matches = (y_samp_np == y_true_np[None, ...]).astype(np.float64)
    match_count = np.sum(matches, axis=0)

    flat = y_samp_np.reshape((s, -1))
    flat_sorted = np.sort(flat, axis=0)
    if s == 1:
        unique_count_flat = np.ones(flat.shape[1], dtype=np.float64)
    else:
        unique_count_flat = 1.0 + \
            np.sum(np.diff(flat_sorted, axis=0) != 0, axis=0, dtype=np.float64)
    unique_count = unique_count_flat.reshape(y_true_np.shape)

    prob = (match_count + alpha) / (s + alpha * np.maximum(unique_count, 1.0))
    return float(np.mean(np.log(np.clip(prob, 1e-12, 1.0))))


def score_log_likelihood_nb2(y_true: jnp.ndarray, y_samples: jnp.ndarray) -> float:
    """Backward-compatible alias.

    Despite the historical name, this score is model-agnostic and only depends on
    discrete predictive samples.
    """
    return score_predictive_log_prob_samples(y_true, y_samples)


def score_crps_counts(
    y_true: jnp.ndarray,
    y_samples: jnp.ndarray,
    *,
    pairwise_block_size: int = 64,
) -> float:
    """Monte Carlo CRPS/energy-score estimate for count forecasts.

    Notes
    -----
    The estimator uses finite posterior predictive samples and therefore has small
    O(1/S) Monte Carlo bias. Pairwise terms are computed blockwise to avoid
    materialising an [S,S,...] tensor.
    """
    y_true_np = np.asarray(y_true, dtype=np.float64)
    y_samp_np = np.asarray(y_samples, dtype=np.float64)
    s = y_samp_np.shape[0]
    if s <= 0:
        raise ValueError("y_samples must contain at least one sample")

    s1 = float(np.mean(np.abs(y_samp_np - y_true_np[None, ...])))

    block = max(int(pairwise_block_size), 1)
    elem_count = int(np.prod(y_samp_np.shape[1:]))
    pairwise_sum = 0.0
    for start in range(0, s, block):
        stop = min(start + block, s)
        chunk = y_samp_np[start:stop]
        pairwise_sum += float(
            np.sum(np.abs(chunk[:, None, ...] - y_samp_np[None, ...])))

    s2 = pairwise_sum / (s * s * elem_count)
    return float(s1 - 0.5 * s2)


def coverage(y_true: jnp.ndarray, q_lo: jnp.ndarray, q_hi: jnp.ndarray) -> float:
    """Empirical interval coverage for observed counts and predictive quantiles."""

    inside = (y_true >= q_lo) & (y_true <= q_hi)
    return float(jnp.mean(inside))


def _spatial_totals(y: np.ndarray) -> np.ndarray:
    """Collapse count arrays to node-level totals for hotspot comparisons."""

    if y.ndim == 3:  # [H, J, M]
        return y.sum(axis=(0, 2))
    if y.ndim == 2:  # [J, M]
        return y.sum(axis=-1)
    if y.ndim == 1:  # [J]
        return y
    raise ValueError(f"Expected y with ndim in {{1,2,3}}, got {y.ndim}")


def hotspot_recall(y_true: jnp.ndarray, y_mean: jnp.ndarray, k: int = 100) -> float:
    """Recall of top-k predicted hotspot nodes against observed top-k nodes."""

    yt = _spatial_totals(np.asarray(y_true, dtype=np.float64))
    yp = _spatial_totals(np.asarray(y_mean, dtype=np.float64))
    n = int(yt.shape[0])
    if n == 0:
        return 0.0
    k_eff = min(max(int(k), 1), n)
    top_true = np.argpartition(np.asarray(yt), -k_eff)[-k_eff:]
    top_pred = np.argpartition(np.asarray(yp), -k_eff)[-k_eff:]
    return float(len(set(top_true).intersection(set(top_pred))) / k_eff)


def aggregate_daily_totals(y: jnp.ndarray) -> pd.DataFrame:
    """Aggregate `[T,J,M]` counts into daily totals with per-mark columns."""
    if y.ndim != 3:
        raise ValueError(f"Expected y with shape [T,J,M], got ndim={y.ndim}")

    total = jnp.sum(y, axis=(1, 2))
    per_mark = jnp.sum(y, axis=1)

    data: dict[str, Any] = {"t": np.arange(
        y.shape[0]), "total": np.asarray(total)}
    for m_idx in range(y.shape[-1]):
        data[f"mark_{m_idx}"] = np.asarray(per_mark[:, m_idx])
    return pd.DataFrame(data)


def aggregate_mark_totals(y: jnp.ndarray) -> pd.DataFrame:
    """Aggregate `[T,J,M]` counts into per-mark totals over all time/nodes."""
    if y.ndim != 3:
        raise ValueError(f"Expected y with shape [T,J,M], got ndim={y.ndim}")

    per_mark = np.asarray(jnp.sum(y, axis=(0, 1)))
    return pd.DataFrame({"mark": np.arange(y.shape[-1]), "total": per_mark})


def rolling_backtest(
    model: MobilityHawkesModel,
    data: CountsTensor,
    *,
    horizon: int = 7,
    step: int = 7,
    min_train: int = 60,
    fit_method: str = "map_ensemble",
    fit_config: FitConfig | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    forecast_samples: int = 200,
    seed: int = 0,
) -> BacktestResult:
    """Run rolling-origin fit/forecast evaluation across temporal splits."""

    splits = data.rolling_origin_splits(
        horizon=horizon, step=step, min_train=min_train)
    forecasts: list[ForecastResult] = []
    rows: list[dict[str, Any]] = []
    fit_kwargs = fit_kwargs or {}

    for i, split in enumerate(splits):
        fit = model.fit(
            split.train,
            method=fit_method,
            config=fit_config,
            seed=seed + i,
            **fit_kwargs,
        )
        fc = model.forecast(
            split.train,
            fit,
            horizon=horizon,
            num_samples=forecast_samples,
            seed=seed + 10_000 + i,
        )
        forecasts.append(fc)

        y_true = split.test.y
        ll = score_predictive_log_prob_samples(y_true, fc.y_samples)
        crps = score_crps_counts(y_true, fc.y_samples)
        cov = coverage(y_true, fc.quantiles[0.05], fc.quantiles[0.95])
        hr = hotspot_recall(y_true, fc.mean)

        rows.append(
            {
                "split": i,
                "origin_index": split.origin_index,
                "log_score": ll,
                "crps": crps,
                "coverage_90": cov,
                "hotspot_recall": hr,
            }
        )

    metrics = pd.DataFrame(rows)
    return BacktestResult(
        splits=splits,
        forecasts=forecasts,
        metrics=metrics,
        meta={
            "horizon": horizon,
            "step": step,
            "min_train": min_train,
            "fit_method": fit_method,
        },
    )
