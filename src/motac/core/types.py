from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    t: int
    lat: float
    lon: float
    cell_id: int
    mark: int | None = None


@dataclass(frozen=True)
class Dataset:
    events: Any
    counts: Any
    meta: dict


@dataclass(frozen=True)
class Substrate:
    grid: Any
    road_graph: Any
    neighbours: Any
    poi_features: Any


class HawkesModel(Protocol):
    def fit(self, dataset: Dataset, substrate: Substrate, cfg: dict) -> Any: ...
    def predict(self, fitted: Any, start_t: int, horizon: int, n_samples: int, seed: int) -> Any: ...
    def loglik(self, fitted: Any, dataset: Dataset) -> float: ...
