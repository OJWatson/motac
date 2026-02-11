"""Backwards-compatible re-export.

Prefer importing from :mod:`motac.loaders.chicago`.
"""

from __future__ import annotations

from .loaders.chicago import ChicagoData, load_y_obs_matrix

__all__ = ["ChicagoData", "load_y_obs_matrix"]
