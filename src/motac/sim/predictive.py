from __future__ import annotations

import numpy as np


def summarize_predictive_paths(
    *,
    paths: np.ndarray,
    q: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict[str, np.ndarray]:
    """Summarize Monte Carlo predictive paths.

    Parameters
    ----------
    paths:
        Array of predictive samples with shape (n_paths, ...).
        The first axis is interpreted as Monte Carlo replicate index.
    q:
        Quantiles to compute along the path axis.

    Returns
    -------
    dict with keys:
      - mean: (...,) predictive mean
      - quantiles: (len(q), ...)
      - q: (len(q),) the quantile levels
    """

    if paths.ndim < 1:
        raise ValueError("paths must have at least one dimension")
    if paths.shape[0] <= 0:
        raise ValueError("paths first dimension must be n_paths > 0")

    qs = np.asarray(q, dtype=float)
    if np.any((qs < 0.0) | (qs > 1.0)):
        raise ValueError("q entries must be in [0,1]")

    mean = paths.mean(axis=0)
    quants = np.quantile(paths, qs, axis=0)

    return {
        "mean": mean,
        "quantiles": quants,
        "q": qs,
    }
