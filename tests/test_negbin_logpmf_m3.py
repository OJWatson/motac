from __future__ import annotations

import numpy as np

from motac.model.likelihood import negbin_logpmf, poisson_logpmf


def test_negbin_logpmf_matches_poisson_at_large_dispersion_approx() -> None:
    rng = np.random.default_rng(0)
    y = rng.poisson(lam=2.0, size=(5,))
    mean = np.full_like(y, 2.0, dtype=float)

    ll_p = poisson_logpmf(y=y, mean=mean).sum()
    ll_nb = negbin_logpmf(y=y, mean=mean, dispersion=1e6).sum()

    assert abs(float(ll_nb - ll_p)) < 1e-3
