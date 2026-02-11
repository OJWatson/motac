from __future__ import annotations

import numpy as np

from .likelihood import negbin_logpmf, poisson_logpmf


def mean_negative_log_likelihood(
    *,
    y: np.ndarray,
    mean: np.ndarray,
    family: str = "poisson",
    dispersion: float | None = None,
) -> float:
    """Mean negative log-likelihood for count observations given predicted means."""

    y = np.asarray(y)
    mean = np.asarray(mean)
    if y.shape != mean.shape:
        raise ValueError("y and mean must have the same shape")

    if family == "poisson":
        nll = -poisson_logpmf(y=y, mean=mean)
        return float(np.mean(nll))

    if family == "negbin":
        if dispersion is None:
            raise ValueError("dispersion must be provided for negbin")
        nll = -negbin_logpmf(y=y, mean=mean, dispersion=float(dispersion))
        return float(np.mean(nll))

    raise ValueError("family must be one of {'poisson','negbin'}")
