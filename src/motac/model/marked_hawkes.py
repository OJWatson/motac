"""Marked Hawkes model scaffolding.

This module is intentionally minimal: it establishes a stable import path and a
small set of types/helpers for future marked Hawkes variants (e.g. event type,
severity, or other per-event marks).

Implementation of marked likelihoods/intensities will land in later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import RoadHawkesDataset


def validate_categorical_marks_matrix(
    marks: np.ndarray,
    *,
    y_obs: np.ndarray,
    n_marks: int | None = None,
) -> np.ndarray:
    """Validate a categorical mark matrix aligned to a binned observation matrix.

    Contract (v1): marks are integer-coded categorical labels with the same shape
    as ``y_obs``.

    Parameters
    ----------
    marks:
        Candidate mark array.
    y_obs:
        Observation count matrix to align against.
    n_marks:
        Optional upper bound on the number of categories. When provided, all mark
        values must satisfy ``0 <= marks < n_marks``.

    Returns
    -------
    numpy.ndarray
        The validated marks as a NumPy array (possibly a view/copy from input).
    """

    m = np.asarray(marks)
    if m.ndim != 2:
        raise ValueError("marks must be 2D (n_cells, n_steps)")

    y = np.asarray(y_obs)
    if m.shape != y.shape:
        raise ValueError(f"marks must match y_obs shape {y.shape}, got shape={m.shape}")

    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError("marks must have an integer dtype for categorical encoding")

    if np.any(m < 0):
        raise ValueError("marks must be non-negative for categorical encoding")

    if n_marks is not None:
        if n_marks <= 0:
            raise ValueError("n_marks must be positive")
        if np.any(m >= n_marks):
            raise ValueError("marks contain values outside [0, n_marks)")

    return m


@dataclass(frozen=True)
class MarkedRoadHawkesDataset:
    """A thin wrapper around :class:`~motac.model.dataset.RoadHawkesDataset`.

    Attributes
    ----------
    base:
        The unmarked road Hawkes dataset.
    marks:
        Mark labels aligned to the binned observation matrix.

        Convention (v1 scaffold): integer-coded categorical marks of shape
        ``(n_cells, n_steps)`` matching ``base.y_obs``.

        The concrete meaning/encoding is model-specific and will be defined by
        later milestones.
    """

    base: RoadHawkesDataset
    marks: np.ndarray

    def __post_init__(self) -> None:  # pragma: no cover
        _ = validate_categorical_marks_matrix(self.marks, y_obs=self.base.y_obs)
