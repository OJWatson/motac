from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def min_travel_time_to_mask(
    *,
    travel_time_s: sp.csr_matrix,
    mask: np.ndarray,
    default: float,
) -> np.ndarray:
    """Compute min travel time from each row to any column where mask is True.

    Parameters
    ----------
    travel_time_s:
        CSR matrix with shape (n_cells, n_cells), where entries are travel times
        in seconds.
    mask:
        Boolean array of shape (n_cells,) indicating target cells.
    default:
        Value used when a row has no reachable target cell.

    Returns
    -------
    out:
        Array of shape (n_cells,) with per-cell minimum travel time.
    """

    if not sp.isspmatrix_csr(travel_time_s):
        travel_time_s = travel_time_s.tocsr()

    n, m = travel_time_s.shape
    if n != m:
        raise ValueError("travel_time_s must be square")

    mask = np.asarray(mask)
    if mask.shape != (n,):
        raise ValueError("mask must have shape (n_cells,)")
    if mask.dtype != bool:
        mask = mask.astype(bool)

    out = np.full((n,), float(default), dtype=float)

    indptr = travel_time_s.indptr
    indices = travel_time_s.indices
    data = travel_time_s.data

    for i in range(n):
        # If the row itself is a target location, distance is zero even if the
        # CSR representation does not explicitly store diagonal zeros.
        if bool(mask[i]):
            out[i] = 0.0
            continue

        start, end = int(indptr[i]), int(indptr[i + 1])
        if start == end:
            continue
        cols = indices[start:end]
        ts = data[start:end]
        sel = mask[cols]
        if np.any(sel):
            out[i] = float(np.min(ts[sel]))

    return out


def min_travel_time_feature_matrix(
    *,
    travel_time_s: sp.csr_matrix,
    masks: dict[str, np.ndarray],
    default: float,
    suffix: str = "min_travel_time_s",
) -> tuple[np.ndarray, list[str]]:
    """Compute a feature matrix of min travel times for multiple target masks.

    Parameters
    ----------
    travel_time_s:
        CSR travel-time matrix.
    masks:
        Dict mapping feature prefix -> boolean mask of target cells.
    default:
        Default value if no target is reachable.
    suffix:
        Suffix appended to each feature name.

    Returns
    -------
    x:
        Array of shape (n_cells, n_features).
    names:
        Feature names in order.
    """

    names: list[str] = []
    cols: list[np.ndarray] = []
    for prefix, mask in masks.items():
        col = min_travel_time_to_mask(
            travel_time_s=travel_time_s,
            mask=mask,
            default=default,
        )
        cols.append(col.reshape(-1, 1))
        names.append(f"{prefix}_{suffix}")

    if len(cols) == 0:
        n = int(travel_time_s.shape[0])
        return np.zeros((n, 0), dtype=float), []

    return np.concatenate(cols, axis=1), names
