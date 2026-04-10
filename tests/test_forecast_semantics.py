import numpy as np
import jax.numpy as jnp

from motac.connectivity import make_grid_stencil
from motac.data import CountsTensor, GridSpec
from motac.infer import FitResult
from motac.model import HawkesModelSpec, MobilityHawkesModel


def _counts_tensor(
    y: jnp.ndarray,
    *,
    marks: tuple[str, ...],
    grid: GridSpec,
    connectivity,
) -> CountsTensor:
    t, j, _ = y.shape
    return CountsTensor(
        y=y,
        t0=np.datetime64("2024-01-01"),
        dt_days=1,
        num_time=t,
        num_nodes=j,
        marks=marks,
        node_coords=None,
        grid=grid,
        connectivity=connectivity,
        covariates={},
    )


def _manual_fit(
    spec: HawkesModelSpec,
    *,
    mu: jnp.ndarray,
    alpha: jnp.ndarray,
    p: jnp.ndarray,
    rho: jnp.ndarray,
    w: jnp.ndarray,
    phi: jnp.ndarray,
) -> FitResult:
    posterior_samples = {
        "mu": mu[None, ...],
        "alpha": alpha[None, ...],
        "p": p[None, ...],
        "rho": rho[None, ...],
        "w": w[None, ...],
        "phi": phi[None, ...],
    }
    return FitResult(
        method="map_ensemble",
        spec=spec,
        posterior_samples=posterior_samples,
        diagnostics={},
    )


def test_forecast_state_initialization_uses_terminal_history_state():
    marks = ("m",)
    grid = GridSpec(shape=(1, 1))
    connectivity = make_grid_stencil(size=1)
    y = jnp.array([[[1.0]], [[5.0]], [[9.0]]], dtype=jnp.float32)
    data = _counts_tensor(y, marks=marks, grid=grid, connectivity=connectivity)

    spec = HawkesModelSpec(
        marks=marks,
        temporal_bases=1,
        observation="poisson",
        normalise_time_kernel=False,
        normalise_spatial_kernel=True,
    )
    model = MobilityHawkesModel(spec)
    fit = _manual_fit(
        spec,
        mu=jnp.array([0.1], dtype=jnp.float32),
        alpha=jnp.array([0.5], dtype=jnp.float32),
        p=jnp.array([[1.0]], dtype=jnp.float32),
        rho=jnp.array([[0.5]], dtype=jnp.float32),
        w=jnp.array([[1.0]], dtype=jnp.float32),
        phi=jnp.array([20.0], dtype=jnp.float32),
    )

    fc = model.forecast(data, fit, horizon=1, num_samples=2000, seed=0)
    mean_est = float(fc.mean[0, 0, 0])

    expected = 0.1 + 0.5 * (0.5 * (0.5 * (0.5 * 0.0 + 1.0) + 5.0) + 9.0)
    buggy_double_count_last = 0.1 + 0.5 * (
        0.5 * (0.5 * (0.5 * (0.5 * 0.0 + 1.0) + 5.0) + 9.0) + 9.0
    )

    assert abs(mean_est - expected) < 0.35
    assert abs(mean_est - expected) < abs(mean_est - buggy_double_count_last)


def test_forecast_temporal_normalisation_flag_changes_gain():
    marks = ("m",)
    grid = GridSpec(shape=(1, 1))
    connectivity = make_grid_stencil(size=1)
    y = jnp.array([[[1.0]], [[5.0]], [[9.0]]], dtype=jnp.float32)
    data = _counts_tensor(y, marks=marks, grid=grid, connectivity=connectivity)

    base_params = dict(
        mu=jnp.array([0.1], dtype=jnp.float32),
        alpha=jnp.array([0.5], dtype=jnp.float32),
        p=jnp.array([[1.0]], dtype=jnp.float32),
        rho=jnp.array([[0.5]], dtype=jnp.float32),
        w=jnp.array([[1.0]], dtype=jnp.float32),
        phi=jnp.array([20.0], dtype=jnp.float32),
    )

    spec_raw = HawkesModelSpec(
        marks=marks,
        temporal_bases=1,
        observation="poisson",
        normalise_time_kernel=False,
    )
    model_raw = MobilityHawkesModel(spec_raw)
    fit_raw = _manual_fit(spec_raw, **base_params)
    fc_raw = model_raw.forecast(
        data, fit_raw, horizon=1, num_samples=2000, seed=1)
    mean_raw = float(fc_raw.mean[0, 0, 0])

    spec_norm = HawkesModelSpec(
        marks=marks,
        temporal_bases=1,
        observation="poisson",
        normalise_time_kernel=True,
    )
    model_norm = MobilityHawkesModel(spec_norm)
    fit_norm = _manual_fit(spec_norm, **base_params)
    fc_norm = model_norm.forecast(
        data, fit_norm, horizon=1, num_samples=2000, seed=1)
    mean_norm = float(fc_norm.mean[0, 0, 0])

    assert mean_raw > mean_norm + 2.0


def test_forecast_spatial_normalisation_override_applies_to_conv_stencil():
    marks = ("m",)
    grid = GridSpec(shape=(2, 2))
    connectivity = make_grid_stencil(kind="moore", size=3, normalise=False)
    y = jnp.ones((2, 4, 1), dtype=jnp.float32)
    data = _counts_tensor(y, marks=marks, grid=grid, connectivity=connectivity)

    params = dict(
        mu=jnp.array([0.1], dtype=jnp.float32),
        alpha=jnp.array([0.25], dtype=jnp.float32),
        p=jnp.array([[1.0]], dtype=jnp.float32),
        rho=jnp.array([[0.0]], dtype=jnp.float32),
        w=jnp.array([[1.0]], dtype=jnp.float32),
        phi=jnp.array([20.0], dtype=jnp.float32),
    )

    spec_raw = HawkesModelSpec(
        marks=marks,
        temporal_bases=1,
        observation="poisson",
        normalise_spatial_kernel=False,
    )
    model_raw = MobilityHawkesModel(spec_raw)
    fit_raw = _manual_fit(spec_raw, **params)
    fc_raw = model_raw.forecast(
        data, fit_raw, horizon=1, num_samples=2000, seed=2)
    mean_raw = float(fc_raw.mean[0].mean())

    spec_norm = HawkesModelSpec(
        marks=marks,
        temporal_bases=1,
        observation="poisson",
        normalise_spatial_kernel=True,
    )
    model_norm = MobilityHawkesModel(spec_norm)
    fit_norm = _manual_fit(spec_norm, **params)
    fc_norm = model_norm.forecast(
        data, fit_norm, horizon=1, num_samples=2000, seed=2)
    mean_norm = float(fc_norm.mean[0].mean())

    assert abs(mean_raw - 1.1) < 0.2
    assert abs(mean_norm - (0.1 + 0.25 * (4.0 / 9.0))) < 0.08
    assert mean_raw > mean_norm + 0.6
