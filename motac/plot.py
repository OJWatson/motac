"""Matplotlib-based plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .data import GridSpec


def plot_intensity_map(grid: GridSpec, values: np.ndarray, *, title: str = "") -> Figure:
    """Render node-level values as a 2D grid heatmap."""

    h, w = grid.shape
    arr = values.reshape(h, w)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Intensity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def plot_forecast_calibration(df_metrics: pd.DataFrame, *, by: str = "horizon") -> Figure:
    """Plot empirical 90% interval coverage against a grouping dimension."""

    fig, ax = plt.subplots(figsize=(7, 4))
    if by in df_metrics.columns:
        grouped = df_metrics.groupby(by)["coverage_90"].mean().reset_index()
        ax.plot(grouped[by], grouped["coverage_90"], marker="o")
        ax.set_xlabel(by)
    else:
        ax.plot(df_metrics.index, df_metrics["coverage_90"], marker="o")
        ax.set_xlabel("index")
    ax.axhline(0.9, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("Coverage (90%)")
    ax.set_title("Forecast calibration")
    return fig


def plot_hotspots(
    grid: GridSpec,
    y_mean: np.ndarray,
    y_true: np.ndarray,
    *,
    k: int = 200,
) -> Figure:
    """Visualize overlap between top-k predicted and observed hotspot nodes."""

    h, w = grid.shape
    if y_mean.ndim == 3:  # [H, J, M]
        pred = y_mean.sum(axis=(0, -1))
    elif y_mean.ndim == 2:  # [J, M]
        pred = y_mean.sum(axis=-1)
    else:
        pred = y_mean.reshape(-1)

    if y_true.ndim == 3:  # [H, J, M]
        true = y_true.sum(axis=(0, -1))
    elif y_true.ndim == 2:  # [J, M]
        true = y_true.sum(axis=-1)
    else:
        true = y_true.reshape(-1)

    n = int(true.shape[0])
    if n == 0:
        raise ValueError("Cannot plot hotspots for empty input")
    k_eff = min(max(int(k), 1), n)

    top_pred = np.argpartition(pred, -k_eff)[-k_eff:]
    top_true = np.argpartition(true, -k_eff)[-k_eff:]

    mask = np.zeros_like(pred)
    mask[top_true] = 1
    mask[top_pred] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mask.reshape(h, w), cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Hotspot overlap (0:none,1:true-only,2:overlap)")
    return fig
