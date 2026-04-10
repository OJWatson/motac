"""Data containers and dataset-shaping helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .connectivity import ConvStencil, EdgeList


@dataclass(frozen=True)
class GridSpec:
    """Spatial lattice metadata.

    Parameters
    ----------
    shape:
        Grid dimensions `(rows, cols)`.
    extent:
        Optional `(lon_min, lon_max, lat_min, lat_max)` bounds used when
        binning point events into grid cells.
    crs:
        Coordinate reference system identifier for `extent`.
    """

    shape: tuple[int, int]
    extent: tuple[float, float, float, float] | None = None
    crs: str | None = "EPSG:4326"


@dataclass
class EventTable:
    """Canonical wrapper for raw event-table style inputs.

    The wrapper stores column mapping metadata and lightweight filtering helpers
    used by dataset adapters.
    """

    df: pd.DataFrame
    time_col: str = "time"
    lat_col: str = "lat"
    lon_col: str = "lon"
    mark_col: str = "mark"
    weight_col: str | None = None

    def validate(self) -> None:
        """Validate that all required columns are present."""

        required = {self.time_col, self.lat_col, self.lon_col, self.mark_col}
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"EventTable missing columns: {missing}")

    def filter_time(self, start: datetime, end: datetime) -> "EventTable":
        """Return a time-windowed copy with UTC-aware timestamp comparison."""

        times = pd.to_datetime(self.df[self.time_col], utc=True)
        mask = (times >= pd.Timestamp(start, tz="UTC")) & (
            times <= pd.Timestamp(end, tz="UTC"))
        return EventTable(
            df=self.df.loc[mask].copy(),
            time_col=self.time_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            mark_col=self.mark_col,
            weight_col=self.weight_col,
        )

    def filter_marks(self, marks: Sequence[str]) -> "EventTable":
        """Return a copy restricted to a supplied mark subset."""

        mask = self.df[self.mark_col].isin(marks)
        return EventTable(
            df=self.df.loc[mask].copy(),
            time_col=self.time_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            mark_col=self.mark_col,
            weight_col=self.weight_col,
        )


@dataclass(frozen=True)
class CountsTensor:
    """Model-ready count tensor with aligned metadata.

    The main array `y` has shape `[T, J, M]` for time, node, and mark axes.
    """

    y: jnp.ndarray
    t0: np.datetime64
    dt_days: int
    num_time: int
    num_nodes: int
    marks: tuple[str, ...]
    node_coords: np.ndarray | None
    grid: GridSpec | None
    connectivity: "ConvStencil | EdgeList | None"
    covariates: dict[str, jnp.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor and metadata consistency invariants."""

        if self.y.ndim != 3:
            raise ValueError(
                f"CountsTensor.y must have shape [T,J,M], got ndim={self.y.ndim}")
        t, j, m = self.y.shape
        if (t, j) != (self.num_time, self.num_nodes):
            raise ValueError(
                f"Shape mismatch: y has {(t, j)}, metadata has {(self.num_time, self.num_nodes)}"
            )
        if m != len(self.marks):
            raise ValueError(
                f"marks length {len(self.marks)} must match y.shape[-1]={m}")

    def with_connectivity(self, connectivity: "ConvStencil | EdgeList") -> "CountsTensor":
        """Return a shallow copy with updated spatial connectivity backend."""

        return CountsTensor(
            y=self.y,
            t0=self.t0,
            dt_days=self.dt_days,
            num_time=self.num_time,
            num_nodes=self.num_nodes,
            marks=self.marks,
            node_coords=self.node_coords,
            grid=self.grid,
            connectivity=connectivity,
            covariates=self.covariates,
        )

    def subset_nodes(
        self,
        keep_nodes: np.ndarray | list[int] | tuple[int, ...],
        *,
        compact: bool = True,
        self_weight: float = 0.0,
    ) -> "CountsTensor":
        if not compact:
            raise ValueError(
                "CountsTensor.subset_nodes currently supports only compact=True")

        keep_arr = np.asarray(keep_nodes)
        if keep_arr.dtype == bool:
            keep_mask = keep_arr.reshape(-1)
            if keep_mask.size != self.num_nodes:
                raise ValueError(
                    f"Boolean keep_nodes mask must have length {self.num_nodes}, got {keep_mask.size}"
                )
            node_index = np.flatnonzero(keep_mask).astype(np.int32)
            if node_index.size == 0:
                raise ValueError("keep_nodes must retain at least one node")
        else:
            node_index = keep_arr.astype(np.int32, copy=False).reshape(-1)
            if node_index.size == 0:
                raise ValueError("keep_nodes must retain at least one node")
            if np.any(node_index < 0) or np.any(node_index >= self.num_nodes):
                raise ValueError("keep_nodes contains out-of-range node ids")
            keep_mask = np.zeros(self.num_nodes, dtype=bool)
            keep_mask[node_index] = True

        y_subset = self.y[:, node_index, :]
        node_coords = None if self.node_coords is None else np.asarray(self.node_coords)[
            node_index]
        covariates = dict(self.covariates)
        covariates["parent_node_index"] = jnp.asarray(
            node_index, dtype=jnp.int32)

        connectivity = self.connectivity
        grid = self.grid
        if connectivity is not None:
            from .connectivity import ConvStencil, EdgeList, subset_edgelist

            if isinstance(connectivity, EdgeList):
                connectivity = subset_edgelist(
                    connectivity,
                    keep_mask,
                    compact=compact,
                    normalise_rows=True,
                    self_weight=self_weight,
                )
            elif isinstance(connectivity, ConvStencil):
                if compact:
                    raise ValueError(
                        "Cannot compact a CountsTensor with ConvStencil connectivity; attach an EdgeList connectivity first"
                    )
            else:
                raise TypeError("Unsupported connectivity type")

        grid = None

        return CountsTensor(
            y=y_subset,
            t0=self.t0,
            dt_days=self.dt_days,
            num_time=self.num_time,
            num_nodes=int(node_index.size),
            marks=self.marks,
            node_coords=node_coords,
            grid=grid,
            connectivity=connectivity,
            covariates=covariates,
        )

    def rolling_origin_splits(
        self,
        horizon: int = 7,
        step: int = 7,
        min_train: int = 60,
    ) -> list["TrainTestSplit"]:
        """Construct rolling-origin train/test splits from this tensor."""

        splits: list[TrainTestSplit] = []
        if self.num_time < (min_train + horizon):
            return splits

        start = min_train
        end = self.num_time - horizon
        for origin in range(start, end + 1, step):
            train_y = self.y[:origin]
            test_y = self.y[origin: origin + horizon]
            train = CountsTensor(
                y=train_y,
                t0=self.t0,
                dt_days=self.dt_days,
                num_time=train_y.shape[0],
                num_nodes=self.num_nodes,
                marks=self.marks,
                node_coords=self.node_coords,
                grid=self.grid,
                connectivity=self.connectivity,
                covariates=self.covariates,
            )
            test = CountsTensor(
                y=test_y,
                t0=self.t0 + np.timedelta64(origin * self.dt_days, "D"),
                dt_days=self.dt_days,
                num_time=test_y.shape[0],
                num_nodes=self.num_nodes,
                marks=self.marks,
                node_coords=self.node_coords,
                grid=self.grid,
                connectivity=self.connectivity,
                covariates=self.covariates,
            )
            # `origin` is the first test index; `origin_index` stores the
            # last index included in the training slice.
            splits.append(TrainTestSplit(
                train=train, test=test, origin_index=origin - 1))
        return splits


@dataclass(frozen=True)
class TrainTestSplit:
    """A single rolling-origin split.

    `origin_index` is the last index in the training window. The test window
    starts at `origin_index + 1`.
    """
    train: CountsTensor
    test: CountsTensor
    origin_index: int
