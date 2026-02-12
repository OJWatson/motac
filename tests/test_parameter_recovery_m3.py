import numpy as np
import scipy.sparse as sp

from motac.model.fit import fit_road_hawkes_mle
from motac.model.road_hawkes import convolved_history_last, exp_travel_time_kernel


def _simulate_road_counts_poisson(
    *,
    travel_time_s: sp.csr_matrix,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    kernel: np.ndarray,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = int(mu.shape[0])
    y = np.zeros((n_cells, n_steps), dtype=int)

    W = exp_travel_time_kernel(travel_time_s=travel_time_s, beta=beta)
    for t in range(n_steps):
        h = convolved_history_last(y=y[:, :t], kernel=kernel)
        lam = mu + float(alpha) * (W @ h)
        lam = np.clip(lam, 0.0, None)
        y[:, t] = rng.poisson(lam=lam)

    return y


def test_m3_parameter_recovery_tiny_substrate_poisson():
    """End-to-end sanity check: fit roughly recovers parameters on tiny synthetic road substrate."""

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

    y = _simulate_road_counts_poisson(
        travel_time_s=travel_time_s,
        mu=mu_true,
        alpha=alpha_true,
        beta=beta_true,
        kernel=kernel,
        n_steps=160,
        seed=7,
    )

    fit = fit_road_hawkes_mle(
        travel_time_s=travel_time_s,
        kernel=kernel,
        y=y,
        family="poisson",
        maxiter=400,
    )

    mu_hat = np.asarray(fit["mu"], dtype=float)
    alpha_hat = float(fit["alpha"])
    beta_hat = float(fit["beta"])

    assert np.isfinite(mu_hat).all()
    assert np.isfinite(alpha_hat)
    assert np.isfinite(beta_hat)

    # Optimisation should improve the objective from default init.
    assert float(fit["loglik"]) >= float(fit["loglik_init"]) - 1e-6

    # Recovery is approximate (small sample) but should be in the right ballpark.
    # We use tolerant checks to avoid flaky CI failures.
    assert np.allclose(mu_hat, mu_true, rtol=0.65, atol=0.25)
    assert np.isclose(alpha_hat, alpha_true, rtol=0.8, atol=0.15)
    assert np.isclose(beta_hat, beta_true, rtol=1.0, atol=5e-4)
