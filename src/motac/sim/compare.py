from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .likelihood import hawkes_loglik_observed_exact, hawkes_loglik_poisson_observed
from .world import World


@dataclass(frozen=True)
class ObservedLoglikComparison:
    """Container for exact vs approximate observed log-likelihood values."""

    ll_exact: float
    ll_poisson_approx: float

    @property
    def delta_exact_minus_approx(self) -> float:
        return float(self.ll_exact - self.ll_poisson_approx)


def compare_observed_loglik_exact_vs_poisson_approx(
    *,
    world: World,
    kernel: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    y_true_for_history: np.ndarray,
    y_true: np.ndarray,
    y_obs: np.ndarray,
    p_detect: float,
    false_rate: float,
) -> ObservedLoglikComparison:
    """Compare exact observed log-likelihood to the cheap Poisson approximation.

    Notes
    -----
    The *exact* observed likelihood conditions on the latent counts ``y_true``
    under the thinning+clutter observation model.

    The Poisson approximation instead treats the observed process as

        y_obs(t) ~ Poisson(p_detect * lambda(t) + false_rate)

    where ``lambda(t)`` is the Hawkes conditional intensity computed from
    ``y_true_for_history``. This is cheap and can be useful for QA/optimisation,
    but is not the exact likelihood under Binomial thinning.
    """

    ll_exact = hawkes_loglik_observed_exact(
        world=world,
        kernel=kernel,
        mu=mu,
        alpha=alpha,
        y_true_for_history=y_true_for_history,
        y_true=y_true,
        y_obs=y_obs,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    ll_approx = hawkes_loglik_poisson_observed(
        world=world,
        kernel=kernel,
        mu=mu,
        alpha=alpha,
        y_true_for_history=y_true_for_history,
        y_obs=y_obs,
        p_detect=p_detect,
        false_rate=false_rate,
    )

    return ObservedLoglikComparison(ll_exact=float(ll_exact), ll_poisson_approx=float(ll_approx))
