"""Experiment configuration structures.

This is a structure-first boundary for configuration objects and experiment
presets.

At M4 it may be minimal; content will be added in later milestones.
"""

from __future__ import annotations

from .acled import AcledEventsCsvConfig
from .chicago import ChicagoRawConfig

__all__ = [
    "AcledEventsCsvConfig",
    "ChicagoRawConfig",
]
