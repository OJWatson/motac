from __future__ import annotations

import numpy as np

from motac.eval import EvalConfig, evaluate_synthetic


def test_evaluate_synthetic_toy_runs_and_returns_expected_keys() -> None:
    cfg = EvalConfig(
        seed=0,
        n_locations=4,
        n_steps_train=30,
        horizon=5,
        n_lags=4,
        beta=1.0,
        mu=0.1,
        alpha=0.5,
        n_paths=50,
        fit_maxiter=100,
        q=(0.1, 0.9),
    )

    out = evaluate_synthetic(cfg)

    assert set(out.keys()) == {"config", "fit", "forecasts", "metrics"}

    metrics = out["metrics"]
    for k in ["nll_test", "rmse", "mae"]:
        assert k in metrics
        assert np.isfinite(metrics[k])

    forecasts = out["forecasts"]
    y_mean = np.asarray(forecasts["y_true_mean"], dtype=float)
    y_q = np.asarray(forecasts["y_true_quantiles"], dtype=float)

    assert y_mean.shape == (cfg.n_locations, cfg.horizon)
    assert y_q.shape == (len(cfg.q), cfg.n_locations, cfg.horizon)
    assert np.all(np.isfinite(y_mean))
    assert np.all(np.isfinite(y_q))

    fit = out["fit"]
    assert len(np.asarray(fit["mu"], dtype=float)) == cfg.n_locations
    assert np.isfinite(float(fit["alpha"]))
    assert np.isfinite(float(fit["beta"]))
