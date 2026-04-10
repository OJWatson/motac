import jax.numpy as jnp

from motac.connectivity import make_grid_stencil
from motac.data import GridSpec
from motac.infer import FitConfig
from motac.model import HawkesModelSpec, MobilityHawkesModel
from motac.simulate import HawkesSimParams, simulate_counts


def _tiny_dataset(seed: int = 0):
    grid = GridSpec(shape=(3, 3))
    marks = ("m1", "m2")
    params = HawkesSimParams(
        mu=0.03,
        alpha_from=jnp.array([0.2, 0.15], dtype=jnp.float32),
        P_from_to=jnp.array([[0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32),
        rho=jnp.array([[0.6, 0.85], [0.55, 0.8]], dtype=jnp.float32),
        w_time=jnp.array([[0.7, 0.3], [0.6, 0.4]], dtype=jnp.float32),
        obs="poisson",
        concentration=20.0,
    )
    data = simulate_counts(
        T=8,
        grid=grid,
        num_nodes=9,
        marks=marks,
        connectivity=make_grid_stencil(size=3),
        params=params,
        seed=seed,
    )
    model = MobilityHawkesModel(HawkesModelSpec(
        marks=marks, temporal_bases=2, observation="poisson"))
    return model, data


def test_svi_fit_smoke():
    model, data = _tiny_dataset(seed=4)
    fit = model.fit(
        data,
        method="svi",
        config=FitConfig(svi_steps=20, svi_num_particles=1),
        seed=5,
    )
    assert fit.method == "svi"
    assert "mu" in fit.posterior_samples
    assert fit.posterior_samples["mu"].ndim == 2
    assert fit.diagnostics["excitation_spectral_radius_mean"] >= 0.0
    assert fit.diagnostics["excitation_spectral_radius_max"] >= fit.diagnostics["excitation_spectral_radius_mean"]
    assert 0.0 <= fit.diagnostics["excitation_subcritical_fraction"] <= 1.0


def test_nuts_small_fit_smoke():
    model, data = _tiny_dataset(seed=7)
    fit = model.fit(
        data,
        method="nuts_small",
        config=FitConfig(nuts_warmup=20, nuts_samples=20,
                         nuts_target_accept=0.75),
        seed=8,
    )
    assert fit.method == "nuts_small"
    assert "alpha" in fit.posterior_samples
    assert fit.posterior_samples["alpha"].shape[-1] == 2
    assert fit.diagnostics["excitation_spectral_radius_mean"] >= 0.0
    assert fit.diagnostics["excitation_spectral_radius_max"] >= fit.diagnostics["excitation_spectral_radius_mean"]
    assert 0.0 <= fit.diagnostics["excitation_subcritical_fraction"] <= 1.0


def test_map_ensemble_retains_multiple_posterior_samples():
    model, data = _tiny_dataset(seed=9)
    fit = model.fit(
        data,
        method="map_ensemble",
        config=FitConfig(map_steps=12, map_ensemble_size=3, map_patience=6),
        seed=10,
    )

    assert fit.posterior_samples["mu"].shape[0] == 3
    assert fit.diagnostics["num_posterior_samples"] == 3
    assert len(fit.diagnostics["ensemble_objectives"]) == 3


def test_map_ensemble_reports_spectral_radius_diagnostics():
    model, data = _tiny_dataset(seed=11)
    fit = model.fit(
        data,
        method="map_ensemble",
        config=FitConfig(map_steps=10, map_ensemble_size=2, map_patience=4),
        seed=12,
    )

    mean_radius = fit.diagnostics["excitation_spectral_radius_mean"]
    max_radius = fit.diagnostics["excitation_spectral_radius_max"]
    subcritical_fraction = fit.diagnostics["excitation_subcritical_fraction"]

    assert mean_radius >= 0.0
    assert max_radius >= mean_radius
    assert 0.0 <= subcritical_fraction <= 1.0
