"""Dataset loaders.

This subpackage groups dataset-specific loading and preprocessing utilities.
"""

from __future__ import annotations

from .acled import AcledData, load_acled_events_csv
from .chicago import ChicagoData, load_y_obs_matrix

__all__ = [
    "AcledData",
    "load_acled_events_csv",
    "ChicagoData",
    "load_y_obs_matrix",
]
