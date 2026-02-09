from __future__ import annotations

import json

import numpy as np


def _assert_forecast_observed_payload(payload: dict, *, n_locations: int, horizon: int) -> None:
    assert set(payload.keys()) == {"meta", "fit", "predict"}

    meta = payload["meta"]
    for k in [
        "n_locations",
        "n_steps_history",
        "horizon",
        "n_paths",
        "n_lags",
        "beta",
        "p_detect",
        "false_rate",
    ]:
        assert k in meta

    assert int(meta["n_locations"]) == n_locations
    assert int(meta["horizon"]) == horizon

    fit = payload["fit"]
    assert "mu" in fit and "alpha" in fit
    assert len(fit["mu"]) == n_locations
    assert isinstance(fit["alpha"], (int, float))

    pred = payload["predict"]
    assert "q" in pred and "mean" in pred and "quantiles" in pred

    mean = np.asarray(pred["mean"], dtype=float)
    quantiles = np.asarray(pred["quantiles"], dtype=float)
    q = pred["q"]

    assert mean.shape == (n_locations, horizon)
    assert quantiles.shape[1:] == (n_locations, horizon)
    assert quantiles.shape[0] == len(q)


def test_forecast_observed_payload_schema_smoke() -> None:
    # Minimal, synthetic payload check (this is *not* an E2E CLI test).
    payload = {
        "meta": {
            "n_locations": 2,
            "n_steps_history": 10,
            "horizon": 3,
            "n_paths": 5,
            "n_lags": 4,
            "beta": 1.0,
            "p_detect": 0.7,
            "false_rate": 0.2,
        },
        "fit": {"mu": [0.1, 0.2], "alpha": 0.3, "loglik": -1.0, "loglik_init": -2.0},
        "predict": {
            "q": [0.05, 0.5, 0.95],
            "mean": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            "quantiles": [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
        },
    }

    # Round-trip through JSON to mimic real usage.
    payload2 = json.loads(json.dumps(payload))
    _assert_forecast_observed_payload(payload2, n_locations=2, horizon=3)
