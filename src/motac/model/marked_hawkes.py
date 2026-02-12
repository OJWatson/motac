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
        marks = np.asarray(self.marks)
        if marks.shape != self.base.y_obs.shape:
            raise ValueError(
                "marks must match base.y_obs shape "
                f"{self.base.y_obs.shape}, got shape={marks.shape}"
            )
