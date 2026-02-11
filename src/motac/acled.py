"""Backwards-compatible re-export.

Prefer importing from :mod:`motac.loaders.acled`.
"""

from __future__ import annotations

from .loaders.acled import AcledData, load_acled_events_csv

__all__ = ["AcledData", "load_acled_events_csv"]
