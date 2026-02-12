"""Backwards-compatible CLI entrypoint module.

Prefer importing from :mod:`motac.cli`.

This file exists to keep import paths stable while the package layout is being
separated (structure-first).
"""

from __future__ import annotations

from .cli import app

__all__ = ["app"]
