from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """Canonical event record.

    Milestone M1 keeps the schema minimal but stable.

    Attributes
    ----------
    t:
        Event time in seconds (float). Use a consistent epoch per dataset.
    lat, lon:
        WGS84 latitude/longitude in decimal degrees.
    mark:
        Optional discrete mark (e.g. event type).
    meta:
        Optional metadata.
    """

    t: float
    lat: float
    lon: float
    mark: str | None = None
    meta: Mapping[str, Any] | None = None
