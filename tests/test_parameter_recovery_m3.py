import numpy as np
import scipy.sparse as sp

from motac.model.validation import run_parameter_recovery_road_hawkes_poisson


def test_m3_parameter_recovery_tiny_substrate_poisson_multiseed() -> None:
    """Multi-seed parameter recovery harness for the M3 road Hawkes Poisson fitter.

    We use tolerant, distributional checks (median + a minimum success count)
    to keep CI stable while still catching regressions.
    """

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

    summary = run_parameter_recovery_road_hawkes_poisson(
        travel_time_s=travel_time_s,
        kernel=kernel,
        mu_true=mu_true,
        alpha_true=alpha_true,
        beta_true=beta_true,
        n_steps=180,
        seeds=[3, 5, 7, 11, 13],
        maxiter=450,
    )

    assert summary.mu_hat.shape == (5, 3)
    assert np.isfinite(summary.mu_hat).all()
    assert np.isfinite(summary.alpha_hat).all()
    assert np.isfinite(summary.beta_hat).all()

    # Optimisation should improve (or at least not worsen) the objective.
    assert (summary.loglik >= summary.loglik_init - 1e-6).all()

    mu_mae = summary.mu_mae_per_seed()
    alpha_err = summary.alpha_abs_err()
    beta_err = summary.beta_abs_err()

    # Median errors should be small on this toy problem.
    assert float(np.median(mu_mae)) <= 0.25
    assert float(np.median(alpha_err)) <= 0.15
    assert float(np.median(beta_err)) <= 6e-4

    # And most seeds should land in a reasonable ballpark.
    assert int(np.sum(mu_mae <= 0.35)) >= 4
    assert int(np.sum(alpha_err <= 0.22)) >= 4
    assert int(np.sum(beta_err <= 9e-4)) >= 4
