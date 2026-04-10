"""Spatial connectivity objects and operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data import GridSpec
from .utils import check_odd


@dataclass(frozen=True)
class EdgeList:
    """Sparse directed graph connectivity for message-passing backends.

    Edges are represented in COO style with aligned `src`, `dst`, and `weight`
    vectors of equal length.
    """

    src: jnp.ndarray
    dst: jnp.ndarray
    weight: jnp.ndarray
    num_nodes: int
    distance: jnp.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConvStencil:
    """Grid convolution stencil used by dense spatial backend operations."""

    kernel: jnp.ndarray
    padding: str = "SAME"
    normalise: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def normalised_kernel(self) -> jnp.ndarray:
        """Return kernel normalized to unit sum when normalization is enabled."""

        if not self.normalise:
            return self.kernel
        denom = jnp.sum(self.kernel)
        return self.kernel / (denom + 1e-8)


def _coerce_node_subset(
    keep_nodes: np.ndarray | list[int] | tuple[int, ...],
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep_arr = np.asarray(keep_nodes)
    if keep_arr.dtype == bool:
        keep_mask = keep_arr.reshape(-1)
        if keep_mask.size != num_nodes:
            raise ValueError(
                f"Boolean keep_nodes mask must have length {num_nodes}, got {keep_mask.size}"
            )
        node_index = np.flatnonzero(keep_mask).astype(np.int32)
        if node_index.size == 0:
            raise ValueError("keep_nodes must retain at least one node")
        return keep_mask.astype(bool, copy=False), node_index

    node_index = keep_arr.astype(np.int32, copy=False).reshape(-1)
    if node_index.size == 0:
        raise ValueError("keep_nodes must retain at least one node")
    if np.any(node_index < 0) or np.any(node_index >= num_nodes):
        raise ValueError("keep_nodes contains out-of-range node ids")

    keep_mask = np.zeros(num_nodes, dtype=bool)
    keep_mask[node_index] = True
    return keep_mask, node_index


def _build_edgelist(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    *,
    num_nodes: int,
    distance: np.ndarray | None,
    normalise_rows: bool,
    meta: dict[str, Any],
) -> EdgeList:
    src_arr = np.asarray(src, dtype=np.int32).reshape(-1)
    dst_arr = np.asarray(dst, dtype=np.int32).reshape(-1)
    weight_arr = np.asarray(weight, dtype=np.float32).reshape(-1)
    distance_arr = None if distance is None else np.asarray(
        distance, dtype=np.float32).reshape(-1)

    if normalise_rows and src_arr.size:
        rowsum = np.bincount(src_arr, weights=weight_arr,
                             minlength=num_nodes).astype(np.float32)
        weight_arr = weight_arr / np.clip(rowsum[src_arr], 1e-8, None)

    return EdgeList(
        src=jnp.asarray(src_arr, dtype=jnp.int32),
        dst=jnp.asarray(dst_arr, dtype=jnp.int32),
        weight=jnp.asarray(weight_arr, dtype=jnp.float32),
        num_nodes=num_nodes,
        distance=None if distance_arr is None else jnp.asarray(
            distance_arr, dtype=jnp.float32),
        meta=dict(meta),
    )


def make_grid_stencil(
    kind: str = "gaussian",
    size: int = 7,
    sigma: float = 1.5,
    *,
    normalise: bool = True,
) -> ConvStencil:
    """Construct a local spatial stencil for gridded convolution backends."""

    check_odd(size, "size")
    ax = jnp.arange(size, dtype=jnp.float32) - size // 2
    xx, yy = jnp.meshgrid(ax, ax, indexing="ij")

    if kind == "gaussian":
        kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    elif kind == "manhattan":
        kernel = 1.0 / (1.0 + jnp.abs(xx) + jnp.abs(yy))
    elif kind == "moore":
        kernel = jnp.ones((size, size), dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown stencil kind: {kind}")

    if normalise:
        kernel = kernel / (jnp.sum(kernel) + 1e-8)

    return ConvStencil(kernel=kernel, padding="SAME", normalise=normalise, meta={"kind": kind})


def make_grid_edgelist(
    grid: GridSpec,
    neighbourhood: str = "moore",
    *,
    weight: str = "uniform",
    directed: bool = True,
    normalise_rows: bool = True,
) -> EdgeList:
    """Build a lattice neighbor graph for edge-list message passing."""

    h, w = grid.shape
    num_nodes = h * w

    if neighbourhood == "moore":
        offsets = [(dr, dc) for dr in (-1, 0, 1)
                   for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    elif neighbourhood in {"von_neumann", "4-neighbour", "4-neighbor"}:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError(f"Unsupported neighbourhood: {neighbourhood}")

    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    d_list: list[float] = []

    seen_undirected: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            src = r * w + c
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                dst = rr * w + cc
                dist = float((dr**2 + dc**2) ** 0.5)
                wt = 1.0 if weight == "uniform" else 1.0 / (dist + 1e-8)
                if directed:
                    src_list.append(src)
                    dst_list.append(dst)
                    w_list.append(wt)
                    d_list.append(dist)
                    continue

                edge_key = (min(src, dst), max(src, dst))
                if edge_key in seen_undirected:
                    continue
                seen_undirected.add(edge_key)

                src_list.append(src)
                dst_list.append(dst)
                w_list.append(wt)
                d_list.append(dist)

                src_list.append(dst)
                dst_list.append(src)
                w_list.append(wt)
                d_list.append(dist)

    src_arr = jnp.asarray(src_list, dtype=jnp.int32)
    dst_arr = jnp.asarray(dst_list, dtype=jnp.int32)
    w_arr = jnp.asarray(w_list, dtype=jnp.float32)

    if normalise_rows:
        rowsum = jax.ops.segment_sum(w_arr, src_arr, num_segments=num_nodes)
        w_arr = w_arr / (rowsum[src_arr] + 1e-8)

    return EdgeList(
        src=src_arr,
        dst=dst_arr,
        weight=w_arr,
        num_nodes=num_nodes,
        distance=jnp.asarray(d_list, dtype=jnp.float32),
        meta={
            "neighbourhood": neighbourhood,
            "weight": weight,
            "directed": directed,
            "normalise_rows": normalise_rows,
        },
    )


def subset_edgelist(
    edges: EdgeList,
    keep_nodes: np.ndarray | list[int] | tuple[int, ...],
    *,
    compact: bool = True,
    normalise_rows: bool = True,
    self_weight: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> EdgeList:
    keep_mask, node_index = _coerce_node_subset(keep_nodes, edges.num_nodes)

    src_full = np.asarray(edges.src, dtype=np.int32)
    dst_full = np.asarray(edges.dst, dtype=np.int32)
    weight_full = np.asarray(edges.weight, dtype=np.float32)
    distance_full = None if edges.distance is None else np.asarray(
        edges.distance, dtype=np.float32)

    edge_mask = keep_mask[src_full] & keep_mask[dst_full]
    src = src_full[edge_mask]
    dst = dst_full[edge_mask]
    weight = weight_full[edge_mask]
    distance = None if distance_full is None else distance_full[edge_mask]

    if compact:
        reindex = np.full(edges.num_nodes, -1, dtype=np.int32)
        reindex[node_index] = np.arange(node_index.size, dtype=np.int32)
        src = reindex[src]
        dst = reindex[dst]
        out_num_nodes = int(node_index.size)
        kept_nodes = np.arange(out_num_nodes, dtype=np.int32)
    else:
        out_num_nodes = int(edges.num_nodes)
        kept_nodes = node_index.astype(np.int32, copy=False)

    if self_weight > 0.0:
        self_nodes = kept_nodes
        src = np.concatenate([src, self_nodes])
        dst = np.concatenate([dst, self_nodes])
        weight = np.concatenate(
            [weight, np.full(self_nodes.shape, float(self_weight), dtype=np.float32)])
        if distance is not None:
            distance = np.concatenate(
                [distance, np.zeros(self_nodes.shape, dtype=np.float32)])

    edge_meta = dict(edges.meta)
    edge_meta.update({
        "subset_compact": compact,
        "subset_num_nodes": int(node_index.size),
        "subset_self_weight": float(self_weight),
    })
    if meta is not None:
        edge_meta.update(meta)

    return _build_edgelist(
        src,
        dst,
        weight,
        num_nodes=out_num_nodes,
        distance=distance,
        normalise_rows=normalise_rows,
        meta=edge_meta,
    )


def make_masked_grid_edgelist(
    grid: GridSpec,
    node_mask: np.ndarray | list[bool] | tuple[bool, ...],
    neighbourhood: str = "moore",
    *,
    weight: str = "uniform",
    directed: bool = True,
    normalise_rows: bool = True,
    compact: bool = True,
    self_weight: float = 0.0,
) -> EdgeList:
    full_edges = make_grid_edgelist(
        grid,
        neighbourhood=neighbourhood,
        weight=weight,
        directed=directed,
        normalise_rows=False,
    )
    return subset_edgelist(
        full_edges,
        node_mask,
        compact=compact,
        normalise_rows=normalise_rows,
        self_weight=self_weight,
        meta={
            "kind": "masked_grid",
            "neighbourhood": neighbourhood,
            "weight": weight,
            "directed": directed,
        },
    )


def apply_conv_stencil(h_2d: jnp.ndarray, stencil: ConvStencil) -> jnp.ndarray:
    """Apply depthwise spatial conv to h_2d [H, W, C].

    Notes
    -----
    The same spatial stencil is shared across all channels. This is intentional:
    each mark/basis channel is diffused by a common mobility kernel.
    """
    lhs = h_2d[None, ...]
    c = h_2d.shape[-1]
    k = stencil.normalised_kernel()
    rhs = jnp.repeat(k[:, :, None, None], repeats=c, axis=3)
    out = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding=stencil.padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=c,
    )
    return out[0]


def apply_edgelist(h: jnp.ndarray, edges: EdgeList) -> jnp.ndarray:
    """Apply sparse message passing to h [J, C]."""
    msg = h[edges.src] * edges.weight[:, None]
    return jax.ops.segment_sum(
        msg,
        edges.dst,
        num_segments=edges.num_nodes,
        indices_are_sorted=False,
        unique_indices=False,
    )


def apply_spatial_backend(
    x: jnp.ndarray,
    *,
    connectivity: ConvStencil | EdgeList,
    grid: GridSpec | None,
    backend: str = "auto",
    normalise_stencil: bool | None = None,
) -> jnp.ndarray:
    """Apply the selected spatial backend to x.

    Parameters
    ----------
    x:
        Tensor with shape [J, C].
    connectivity:
        Spatial backend object.
    grid:
        Required for convolution backend to reshape J -> [H, W].
    backend:
        One of {"auto", "conv", "edges", "bcoo"}.
    """
    chosen = backend
    if chosen == "auto":
        if isinstance(connectivity, ConvStencil):
            chosen = "conv"
        elif isinstance(connectivity, EdgeList):
            chosen = "edges"
        else:
            raise TypeError("Unsupported connectivity backend")

    if chosen == "conv":
        if not isinstance(connectivity, ConvStencil):
            raise TypeError("backend='conv' requires ConvStencil connectivity")
        if grid is None:
            raise ValueError("Conv backend requires GridSpec")
        stencil = connectivity
        if normalise_stencil is not None and normalise_stencil != connectivity.normalise:
            stencil = ConvStencil(
                kernel=connectivity.kernel,
                padding=connectivity.padding,
                normalise=normalise_stencil,
                meta=connectivity.meta,
            )
        h, w = grid.shape
        return apply_conv_stencil(x.reshape(h, w, -1), stencil).reshape(x.shape)

    if chosen == "edges":
        if not isinstance(connectivity, EdgeList):
            raise TypeError("backend='edges' requires EdgeList connectivity")
        return apply_edgelist(x, connectivity)

    if chosen == "bcoo":
        raise NotImplementedError("BCOO backend is deferred in v1")

    raise ValueError(f"Unknown backend mode: {backend}")
