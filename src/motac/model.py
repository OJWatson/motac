"""Core model specification and high-level model API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan as numpyro_scan

from .connectivity import apply_spatial_backend
from .data import CountsTensor

if TYPE_CHECKING:
    from .forecast import ForecastResult
    from .infer import FitConfig, FitResult


@dataclass(frozen=True)
class HawkesModelSpec:
    """Static configuration for the mobility-constrained marked Hawkes model.

    The spec is shared across simulation, fitting, and forecasting so that
    temporal/spatial normalization semantics remain aligned across workflows.
    """

    marks: tuple[str, ...]
    temporal_bases: int = 2
    use_baseline_covariates: bool = False
    observation: str = "nb2"
    alpha_max: float = 0.95
    normalise_time_kernel: bool = False
    normalise_spatial_kernel: bool = True
    backend: str = "auto"


def unpack_unconstrained(theta: dict[str, jnp.ndarray], spec: HawkesModelSpec) -> dict[str, jnp.ndarray]:
    """Map unconstrained optimization parameters to constrained model parameters."""

    mu = jax.nn.softplus(theta["mu_raw"]) + 1e-8
    alpha = spec.alpha_max * jax.nn.sigmoid(theta["alpha_raw"])
    p = jax.nn.softmax(theta["p_raw"], axis=-1)
    rho = jax.nn.sigmoid(theta["rho_raw"])
    w = jax.nn.softmax(theta["w_raw"], axis=-1)
    phi = jax.nn.softplus(theta["phi_raw"]) + 1e-6
    return {"mu": mu, "alpha": alpha, "p": p, "rho": rho, "w": w, "phi": phi}


def _logprob_obs(y_t: jnp.ndarray, lam_t: jnp.ndarray, phi: jnp.ndarray, observation: str) -> jnp.ndarray:
    """Evaluate elementwise observation log-probabilities for one time step."""

    if observation == "poisson":
        return dist.Poisson(lam_t).log_prob(y_t)
    if observation == "nb2":
        phi_jm = jnp.broadcast_to(phi[None, :], lam_t.shape)
        return dist.NegativeBinomial2(mean=lam_t, concentration=phi_jm).log_prob(y_t)
    raise ValueError(f"Unsupported observation: {observation}")


def effective_excitation_matrix(
    params: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
) -> jnp.ndarray:
    """Effective mark-to-mark excitation matrix.

    For `normalise_time_kernel=False`, each source mark is scaled by the total
    temporal memory mass `sum_b w_{m,b}/(1-rho_{m,b})`. When
    `normalise_time_kernel=True`, temporal kernels integrate to one.
    """
    g = params["alpha"][:, None] * params["p"]
    if spec.normalise_time_kernel:
        temporal_mass = jnp.ones_like(params["alpha"])
    else:
        rho = jnp.clip(params["rho"], 1e-6, 1.0 - 1e-6)
        temporal_mass = jnp.sum(params["w"] / (1.0 - rho), axis=-1)
    return temporal_mass[:, None] * g


def excitation_spectral_radius(
    params: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
) -> float:
    """Spectral radius of the effective excitation matrix."""
    eff = effective_excitation_matrix(params, spec)
    eigvals = jnp.linalg.eigvals(eff.astype(jnp.complex64))
    return float(jnp.max(jnp.abs(eigvals)))


def _loglik_core(
    y: jnp.ndarray,
    data: CountsTensor,
    params: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
    scan_impl: Callable[..., tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]],
) -> jnp.ndarray:
    """Shared recurrence likelihood core used by both JAX and NumPyro scan paths."""

    _, j, m = y.shape
    b = spec.temporal_bases

    g = params["alpha"][:, None] * params["p"]
    mu_jm = jnp.broadcast_to(params["mu"][None, :], (j, m))
    temporal_scale = jnp.where(
        spec.normalise_time_kernel, 1.0 - params["rho"], 1.0)

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], y_t: jnp.ndarray):
        h_state, y_prev = carry
        h_state = params["rho"][None, :, :] * h_state + y_prev[:, :, None]
        h_flat = h_state.reshape(j, m * b)
        h_spatial = apply_spatial_backend(
            h_flat,
            connectivity=data.connectivity,
            grid=data.grid,
            backend=spec.backend,
            normalise_stencil=spec.normalise_spatial_kernel,
        ).reshape(j, m, b)
        mixed = jnp.sum(
            params["w"][None, :, :] * temporal_scale[None, :, :] * h_spatial,
            axis=-1,
        )
        lam_t = jnp.clip(mu_jm + mixed @ g, 1e-8, 1e12)
        ll = _logprob_obs(y_t, lam_t, params["phi"], spec.observation).sum()
        return (h_state, y_t), ll

    init = (jnp.zeros((j, m, b), dtype=jnp.float32), y[0])
    (_, _), ll_series = scan_impl(step, init, y[1:])

    lam0 = mu_jm
    ll0 = _logprob_obs(y[0], lam0, params["phi"], spec.observation).sum()
    return ll0 + ll_series.sum()


def loglik_scan(
    y: jnp.ndarray,
    data: CountsTensor,
    params: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
) -> jnp.ndarray:
    """Compute full data log-likelihood using `jax.lax.scan` recurrence."""

    return _loglik_core(y=y, data=data, params=params, spec=spec, scan_impl=jax.lax.scan)


def numpyro_model(data: CountsTensor, spec: HawkesModelSpec) -> None:
    """NumPyro model definition used by SVI and NUTS inference backends."""

    y = data.y.astype(jnp.float32)
    _, _, m = y.shape
    b = spec.temporal_bases

    mu_raw = numpyro.sample("mu_raw", dist.Normal(jnp.full((m,), -3.0), 1.0))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(jnp.zeros((m,)), 1.0))
    p_raw = numpyro.sample("p_raw", dist.Normal(jnp.zeros((m, m)), 1.0))
    rho_raw = numpyro.sample("rho_raw", dist.Normal(jnp.zeros((m, b)), 1.0))
    w_raw = numpyro.sample("w_raw", dist.Normal(jnp.zeros((m, b)), 1.0))
    phi_raw = numpyro.sample("phi_raw", dist.Normal(jnp.full((m,), 3.0), 0.5))

    params = unpack_unconstrained(
        {
            "mu_raw": mu_raw,
            "alpha_raw": alpha_raw,
            "p_raw": p_raw,
            "rho_raw": rho_raw,
            "w_raw": w_raw,
            "phi_raw": phi_raw,
        },
        spec,
    )

    ll = _loglik_core(y=y, data=data, params=params,
                      spec=spec, scan_impl=numpyro_scan)
    numpyro.factor("obs_ll", ll)


class MobilityHawkesModel:
    """User-facing model wrapper exposing unified fit and forecast methods."""

    def __init__(self, spec: HawkesModelSpec):
        self.spec = spec

    def fit(
        self,
        data: CountsTensor,
        *,
        method: str = "map_ensemble",
        config: FitConfig | None = None,
        seed: int = 0,
    ) -> FitResult:
        """Fit model parameters with selected backend (`map_ensemble`, `svi`, `nuts_small`)."""

        from .infer import FitConfig, fit_model

        cfg = config or FitConfig()
        return fit_model(data=data, spec=self.spec, method=method, config=cfg, seed=seed)

    def forecast(
        self,
        data: CountsTensor,
        fit: FitResult,
        *,
        horizon: int = 7,
        num_samples: int = 200,
        seed: int = 0,
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> ForecastResult:
        """Run posterior predictive rollout from a fitted model state."""

        from .forecast import forecast_from_fit

        return forecast_from_fit(
            data=data,
            spec=self.spec,
            fit=fit,
            horizon=horizon,
            num_samples=num_samples,
            seed=seed,
            quantile_levels=quantile_levels,
        )
