import numpy as np
import scipy.sparse as sp

from motac.model.simulate import simulate_road_hawkes_counts
from motac.model.workflows import fit_forecast_road_hawkes_mle


def test_m3_2_fit_forecast_workflow_smoke_poisson() -> None:
    # Tiny 3-cell substrate with travel times (seconds).
    tt = np.array(
        [
            [0.0, 300.0, 900.0],
            [300.0, 0.0, 600.0],
            [900.0, 600.0, 0.0],
        ],
        dtype=float,
    )
    travel_time_s = sp.csr_matrix(tt)

    kernel = np.array([0.55, 0.30, 0.15], dtype=float)

    mu_true = np.array([0.6, 0.9, 0.5], dtype=float)
    alpha_true = 0.35
    beta_true = 1.2e-3

    y = simulate_road_hawkes_counts(
        travel_time_s=travel_time_s,
        mu=mu_true,
        alpha=alpha_true,
        beta=beta_true,
        kernel=kernel,
        T=120,
        seed=7,
        family="poisson",
    )

    out = fit_forecast_road_hawkes_mle(
        travel_time_s=travel_time_s,
        kernel=kernel,
        y=y,
        horizon=5,
        family="poisson",
        maxiter=300,
    )

    fit = out["fit"]
    lam = np.asarray(out["lam_forecast"], dtype=float)

    assert lam.shape == (3, 5)
    assert np.isfinite(lam).all()
    assert (lam >= 0.0).all()

    # The workflow should return the standard fit keys.
    assert "mu" in fit
    assert "alpha" in fit
    assert "beta" in fit
    assert float(fit["loglik"]) >= float(fit["loglik_init"]) - 1e-6
