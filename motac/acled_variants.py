from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.path import Path as MplPath

from .connectivity import make_grid_stencil, make_masked_grid_edgelist
from .data import CountsTensor, EventTable, GridSpec
from .datasets import acled_to_counts


@dataclass(frozen=True)
class AcledSpatialVariant:
    name: str
    counts: CountsTensor
    grid: GridSpec
    shape: tuple[int, int]
    extent: tuple[float, float, float, float]
    mask: np.ndarray
    mask_flat: np.ndarray
    parent_node_index: np.ndarray
    cell_size_m: float
    gaza_only: bool


def _build_projected_grid(
    projected_extent: tuple[float, float, float, float],
    cell_size_m: float,
) -> tuple[GridSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = projected_extent
    n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size_m)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size_m)))
    grid = GridSpec(shape=(n_rows, n_cols), extent=projected_extent, crs="LOCAL_GAZA_EQUIRECT_M")

    x_edges = np.linspace(x_min, x_max, n_cols + 1)
    y_edges = np.linspace(y_min, y_max, n_rows + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_xx, grid_yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    return grid, x_edges, y_edges, grid_xx, grid_yy


def build_acled_spatial_variant(
    events: EventTable,
    *,
    marks: Sequence[str],
    polygon_xy: np.ndarray,
    projected_extent: tuple[float, float, float, float],
    cell_size_m: float,
    dt_days: int = 14,
    gaza_only: bool = False,
    self_weight: float = 0.2,
    stencil_kind: str = "gaussian",
    stencil_size: int = 3,
    stencil_sigma: float = 1.0,
    neighbourhood: str = "moore",
    name: str,
) -> AcledSpatialVariant:
    grid, _, _, grid_xx, grid_yy = _build_projected_grid(projected_extent, cell_size_m)
    polygon_path = MplPath(np.asarray(polygon_xy, dtype=float))
    mask = polygon_path.contains_points(
        np.column_stack([grid_xx.reshape(-1), grid_yy.reshape(-1)]),
        radius=1e-9,
    ).reshape(grid.shape)
    mask_flat = mask.reshape(-1)

    counts_base = acled_to_counts(events, grid, marks=marks, dt_days=dt_days)
    if gaza_only:
        masked_graph = make_masked_grid_edgelist(
            grid,
            mask_flat,
            neighbourhood=neighbourhood,
            weight="uniform",
            directed=True,
            normalise_rows=True,
            compact=False,
            self_weight=self_weight,
        )
        counts = counts_base.with_connectivity(masked_graph).subset_nodes(mask_flat, compact=True)
        parent_node_index = np.asarray(counts.covariates["parent_node_index"], dtype=np.int32)
    else:
        counts = counts_base.with_connectivity(
            make_grid_stencil(kind=stencil_kind, size=stencil_size, sigma=stencil_sigma)
        )
        parent_node_index = np.arange(counts.num_nodes, dtype=np.int32)

    return AcledSpatialVariant(
        name=name,
        counts=counts,
        grid=grid,
        shape=grid.shape,
        extent=projected_extent,
        mask=mask,
        mask_flat=mask_flat,
        parent_node_index=parent_node_index,
        cell_size_m=float(cell_size_m),
        gaza_only=bool(gaza_only),
    )


def build_standard_acled_variants(
    events: EventTable,
    *,
    marks: Sequence[str],
    polygon_xy: np.ndarray,
    projected_extent: tuple[float, float, float, float],
    dt_days: int = 14,
    self_weight: float = 0.2,
) -> dict[str, AcledSpatialVariant]:
    return {
        "rect_1km": build_acled_spatial_variant(
            events,
            marks=marks,
            polygon_xy=polygon_xy,
            projected_extent=projected_extent,
            cell_size_m=1_000.0,
            dt_days=dt_days,
            gaza_only=False,
            self_weight=self_weight,
            name="rect_1km",
        ),
        "gaza_1km": build_acled_spatial_variant(
            events,
            marks=marks,
            polygon_xy=polygon_xy,
            projected_extent=projected_extent,
            cell_size_m=1_000.0,
            dt_days=dt_days,
            gaza_only=True,
            self_weight=self_weight,
            name="gaza_1km",
        ),
        "gaza_2km": build_acled_spatial_variant(
            events,
            marks=marks,
            polygon_xy=polygon_xy,
            projected_extent=projected_extent,
            cell_size_m=2_000.0,
            dt_days=dt_days,
            gaza_only=True,
            self_weight=self_weight,
            name="gaza_2km",
        ),
    }


def flatten_variant_counts(variant: AcledSpatialVariant, y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 3:
        return arr if variant.gaza_only else arr[:, variant.mask_flat, :]
    if arr.ndim == 4:
        return arr if variant.gaza_only else arr[:, :, variant.mask_flat, :]
    raise ValueError(f"Unexpected tensor rank: {arr.ndim}")


def variant_node_values_to_map(variant: AcledSpatialVariant, node_values: np.ndarray) -> np.ndarray:
    values = np.asarray(node_values, dtype=float).reshape(-1)
    if values.shape[0] != variant.counts.num_nodes:
        raise ValueError(
            f"Expected {variant.counts.num_nodes} node values, got {values.shape[0]}"
        )
    full = np.full(int(np.prod(variant.shape)), np.nan, dtype=float)
    full[variant.parent_node_index] = values
    out = full.reshape(variant.shape)
    out[~variant.mask] = np.nan
    return out


def aggregate_variant_counts_to_map(variant: AcledSpatialVariant, y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 3:
        node_values = arr.sum(axis=(0, 2))
    elif arr.ndim == 2:
        node_values = arr.sum(axis=-1)
    elif arr.ndim == 1:
        node_values = arr
    else:
        raise ValueError(f"Unexpected tensor rank: {arr.ndim}")
    return variant_node_values_to_map(variant, node_values)


def summarise_variant_zero_mass(variant: AcledSpatialVariant) -> dict[str, float]:
    y = np.asarray(variant.counts.y)
    active = flatten_variant_counts(variant, y)
    node_totals = active.sum(axis=(0, 2))
    return {
        "variant": variant.name,
        "cell_size_m": float(variant.cell_size_m),
        "gaza_only": float(variant.gaza_only),
        "grid_rows": float(variant.shape[0]),
        "grid_cols": float(variant.shape[1]),
        "num_grid_cells": float(np.prod(variant.shape)),
        "num_gaza_cells": float(variant.mask.sum()),
        "num_model_nodes": float(variant.counts.num_nodes),
        "zero_fraction": float(np.mean(active == 0.0)),
        "nonzero_fraction": float(np.mean(active > 0.0)),
        "always_zero_node_fraction": float(np.mean(node_totals == 0.0)),
        "total_events": float(y.sum()),
    }
