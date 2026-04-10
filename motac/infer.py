"""Inference backends: MAP ensemble, SVI, and small NUTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import summary as numpyro_summary
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoNormal

from .data import CountsTensor
from .model import (
    HawkesModelSpec,
    excitation_spectral_radius,
    loglik_scan,
    numpyro_model,
    unpack_unconstrained,
)


LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class FitConfig:
    """Configuration for all inference backends.

    The fields are grouped by backend prefix (`map_*`, `svi_*`, `nuts_*`).
    """

    map_steps: int = 1500
    map_lr: float = 2e-2
    map_clip_norm: float = 1.0
    map_ensemble_size: int = 8
    map_patience: int = 200
    svi_steps: int = 3000
    svi_lr: float = 5e-3
    svi_guide: str = "AutoDiagonalNormal"
    svi_elbo: str = "AutoContinuousELBO"
    svi_num_particles: int = 1
    svi_posterior_samples: int = 200
    nuts_warmup: int = 500
    nuts_samples: int = 500
    nuts_chains: int = 1
    nuts_target_accept: float = 0.8
    nuts_max_tj: int = 35_000


@dataclass(frozen=True)
class FitResult:
    """Unified output container for fitted model parameters and diagnostics."""

    method: str
    spec: HawkesModelSpec
    posterior_samples: dict[str, jnp.ndarray]
    diagnostics: dict[str, Any]


def _transform_raw_samples(
    raw_samples: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
) -> dict[str, jnp.ndarray]:
    """Transform unconstrained posterior samples into model-native constrained parameters."""
    mu_raw = raw_samples["mu_raw"]
    alpha_raw = raw_samples["alpha_raw"]
    p_raw = raw_samples["p_raw"]
    rho_raw = raw_samples["rho_raw"]
    w_raw = raw_samples["w_raw"]
    phi_raw = raw_samples["phi_raw"]

    mu = jax.nn.softplus(mu_raw) + 1e-8
    alpha = spec.alpha_max * jax.nn.sigmoid(alpha_raw)
    p = jax.nn.softmax(p_raw, axis=-1)
    rho = jax.nn.sigmoid(rho_raw)
    w = jax.nn.softmax(w_raw, axis=-1)
    phi = jax.nn.softplus(phi_raw) + 1e-6

    return {
        "mu": mu,
        "alpha": alpha,
        "p": p,
        "rho": rho,
        "w": w,
        "phi": phi,
    }


def _init_theta(m: int, b: int, seed: int) -> dict[str, jnp.ndarray]:
    """Initialize unconstrained parameters near weak-prior centers."""

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        "mu_raw": -3.0 + 0.1 * jax.random.normal(k1, (m,)),
        "alpha_raw": 0.1 * jax.random.normal(k2, (m,)),
        "p_raw": 0.1 * jax.random.normal(k3, (m, m)),
        "rho_raw": 0.1 * jax.random.normal(k4, (m, b)),
        "w_raw": 0.1 * jax.random.normal(k5, (m, b)),
        "phi_raw": 3.0 + 0.1 * jax.random.normal(k6, (m,)),
    }


def _logprior(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Gaussian log-prior over unconstrained parameters."""

    return (
        dist_normal_logpdf(theta["mu_raw"], -3.0, 1.0).sum()
        + dist_normal_logpdf(theta["alpha_raw"], 0.0, 1.0).sum()
        + dist_normal_logpdf(theta["p_raw"], 0.0, 1.0).sum()
        + dist_normal_logpdf(theta["rho_raw"], 0.0, 1.0).sum()
        + dist_normal_logpdf(theta["w_raw"], 0.0, 1.0).sum()
        + dist_normal_logpdf(theta["phi_raw"], 3.0, 0.5).sum()
    )


def dist_normal_logpdf(x: jnp.ndarray, loc: float, scale: float) -> jnp.ndarray:
    """Elementwise Normal log-density for scalar location/scale."""

    z = (x - loc) / scale
    return -0.5 * (LOG_2PI + 2.0 * jnp.log(scale) + z**2)


def _map_objective(
    theta: dict[str, jnp.ndarray],
    data: CountsTensor,
    spec: HawkesModelSpec,
) -> jnp.ndarray:
    """Negative log-posterior objective used for MAP optimization."""

    params = unpack_unconstrained(theta, spec)
    ll = loglik_scan(data.y.astype(jnp.float32), data, params, spec)
    lp = _logprior(theta)
    return -(ll + lp)


def _spectral_radius_diagnostics(
    posterior_samples: dict[str, jnp.ndarray],
    spec: HawkesModelSpec,
) -> dict[str, Any]:
    """Summarize effective excitation stability over posterior samples."""

    num = int(posterior_samples["mu"].shape[0])
    radii = np.array(
        [
            excitation_spectral_radius(
                {k: v[i] for k, v in posterior_samples.items()},
                spec,
            )
            for i in range(num)
        ],
        dtype=np.float64,
    )
    return {
        "excitation_spectral_radius_mean": float(np.mean(radii)),
        "excitation_spectral_radius_max": float(np.max(radii)),
        "excitation_subcritical_fraction": float(np.mean(radii < 1.0)),
    }


def _flatten_summary_metric(
    summary_dict: dict[str, dict[str, Any]],
    metric: str,
) -> np.ndarray:
    """Flatten one NumPyro summary metric across all named parameters."""

    arrays: list[np.ndarray] = []
    for stats in summary_dict.values():
        if metric not in stats:
            continue
        arrays.append(np.asarray(stats[metric], dtype=np.float64).reshape(-1))
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays, axis=0)


def _fit_map(data: CountsTensor, spec: HawkesModelSpec, config: FitConfig, seed: int) -> FitResult:
    """Run MAP-ensemble fitting with early stopping per restart."""

    m = len(spec.marks)
    b = spec.temporal_bases

    tx = optax.chain(optax.clip_by_global_norm(
        config.map_clip_norm), optax.adam(config.map_lr))

    objective_and_grad = jax.value_and_grad(
        lambda th: _map_objective(th, data, spec))

    @jax.jit
    def step_fn(th: dict[str, jnp.ndarray], st: optax.OptState):
        loss, grads = objective_and_grad(th)
        updates, st2 = tx.update(grads, st, th)
        th2 = optax.apply_updates(th, updates)
        return th2, st2, loss

    best_obj = np.inf
    best_trace: list[float] = []
    ensemble_thetas: list[dict[str, jnp.ndarray]] = []
    ensemble_objectives: list[float] = []

    for i in range(config.map_ensemble_size):
        theta = _init_theta(m, b, seed + i)
        opt_state = tx.init(theta)

        trace: list[float] = []
        no_improve = 0
        local_best = np.inf
        local_best_theta = theta

        for _ in range(config.map_steps):
            theta, opt_state, loss = step_fn(theta, opt_state)
            loss_val = float(loss)
            trace.append(loss_val)
            if loss_val < local_best - 1e-6:
                local_best = loss_val
                local_best_theta = theta
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= config.map_patience:
                break

        ensemble_thetas.append(local_best_theta)
        ensemble_objectives.append(local_best)

        if local_best < best_obj:
            best_obj = local_best
            best_trace = trace

    if not ensemble_thetas:
        raise RuntimeError("MAP ensemble produced no solutions")

    params_list = [unpack_unconstrained(th, spec) for th in ensemble_thetas]
    posterior_samples = {
        k: jnp.stack([params[k] for params in params_list], axis=0)
        for k in params_list[0]
    }

    best_idx = int(np.argmin(np.asarray(ensemble_objectives)))
    return FitResult(
        method="map_ensemble",
        spec=spec,
        posterior_samples=posterior_samples,
        diagnostics={
            "best_objective": best_obj,
            "best_restart_index": best_idx,
            "objective_trace": best_trace,
            "ensemble_objectives": ensemble_objectives,
            "num_posterior_samples": len(ensemble_thetas),
            **_spectral_radius_diagnostics(posterior_samples, spec),
        },
    )


def _build_guide(name: str):
    """Resolve supported NumPyro autoguide class from configuration name."""

    if name == "AutoNormal":
        return AutoNormal
    return AutoDiagonalNormal


def _build_svi_loss(config: FitConfig):
    """Construct SVI loss object with compatibility fallback logic."""

    if config.svi_elbo == "Trace_ELBO":
        return Trace_ELBO(num_particles=config.svi_num_particles)
    if config.svi_elbo == "AutoContinuousELBO":
        try:
            from numpyro.infer import AutoContinuousELBO

            return AutoContinuousELBO(num_particles=config.svi_num_particles)
        except Exception:
            return Trace_ELBO(num_particles=config.svi_num_particles)
    raise ValueError(f"Unsupported svi_elbo: {config.svi_elbo}")


def _fit_svi(data: CountsTensor, spec: HawkesModelSpec, config: FitConfig, seed: int) -> FitResult:
    """Run stochastic variational inference and return transformed samples."""

    guide_cls = _build_guide(config.svi_guide)
    guide = guide_cls(numpyro_model)
    loss = _build_svi_loss(config)

    svi = SVI(
        numpyro_model,
        guide,
        optim=numpyro.optim.Adam(config.svi_lr),
        loss=loss,
    )

    key = jax.random.PRNGKey(seed)
    result = svi.run(key, config.svi_steps, data=data,
                     spec=spec, progress_bar=False)
    sample_key = jax.random.fold_in(key, 1)
    posterior = guide.sample_posterior(
        sample_key,
        result.params,
        sample_shape=(config.svi_posterior_samples,),
        data=data,
        spec=spec,
    )
    transformed = _transform_raw_samples(posterior, spec)

    return FitResult(
        method="svi",
        spec=spec,
        posterior_samples=transformed,
        diagnostics={
            "losses": result.losses.tolist(),
            **_spectral_radius_diagnostics(transformed, spec),
        },
    )


def _fit_nuts_small(data: CountsTensor, spec: HawkesModelSpec, config: FitConfig, seed: int) -> FitResult:
    """Run a small-NUTS posterior approximation for tiny debugging problems."""

    t, j, _ = data.y.shape
    if t * j > config.nuts_max_tj:
        raise ValueError(
            f"nuts_small is restricted to problems with t*j <= {config.nuts_max_tj}"
        )

    kernel = NUTS(numpyro_model, target_accept_prob=config.nuts_target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=config.nuts_warmup,
        num_samples=config.nuts_samples,
        num_chains=config.nuts_chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data=data, spec=spec)
    raw_samples_grouped = mcmc.get_samples(group_by_chain=True)
    transformed_grouped = _transform_raw_samples(raw_samples_grouped, spec)
    transformed = {
        key: value.reshape((-1,) + tuple(value.shape[2:]))
        for key, value in transformed_grouped.items()
    }
    summary_dict = numpyro_summary(transformed_grouped, group_by_chain=True)
    r_hat = _flatten_summary_metric(summary_dict, "r_hat")
    n_eff = _flatten_summary_metric(summary_dict, "n_eff")
    extra_fields = mcmc.get_extra_fields(group_by_chain=False)
    diverging = np.asarray(extra_fields.get("diverging", []), dtype=bool)

    return FitResult(
        method="nuts_small",
        spec=spec,
        posterior_samples=transformed,
        diagnostics={
            "num_samples": config.nuts_samples,
            "num_warmup": config.nuts_warmup,
            "num_chains": config.nuts_chains,
            "target_accept": config.nuts_target_accept,
            "max_tj": config.nuts_max_tj,
            "num_divergences": int(diverging.sum()) if diverging.size else 0,
            "divergence_fraction": float(diverging.mean()) if diverging.size else 0.0,
            "summary_r_hat_max": float(np.nanmax(r_hat)) if r_hat.size else float("nan"),
            "summary_r_hat_median": float(np.nanmedian(r_hat)) if r_hat.size else float("nan"),
            "summary_n_eff_min": float(np.nanmin(n_eff)) if n_eff.size else float("nan"),
            "mcmc_summary": summary_dict,
            "posterior_samples_grouped": transformed_grouped,
            **_spectral_radius_diagnostics(transformed, spec),
        },
    )


def fit_model(
    data: CountsTensor,
    spec: HawkesModelSpec,
    method: str,
    config: FitConfig,
    seed: int,
) -> FitResult:
    """Dispatch to configured inference backend and return a `FitResult`."""

    if method == "map_ensemble":
        return _fit_map(data, spec, config, seed)
    if method == "svi":
        return _fit_svi(data, spec, config, seed)
    if method == "nuts_small":
        return _fit_nuts_small(data, spec, config, seed)
    raise ValueError(f"Unknown fit method: {method}")
