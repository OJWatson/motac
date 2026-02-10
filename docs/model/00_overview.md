# Model overview

`motac` targets the **Road-Constrained Spatio-Temporal Hawkes** model described in the project PDF.

At a high level, the model operates on discretised spatial cells (nodes) laid over a road network substrate.
Let $y_{j,t}$ be the count of events in cell $j$ during discrete time bin $t$.

## Substrate (road constraints)

The road network induces a travel-time distance $d_{jk}$ between cells $j$ and $k$ (e.g. shortest path travel time).
We typically only consider neighbours $k \in \mathcal{N}(j)$ within some travel-time cutoff.

**TODO (M1):** document and stabilise the substrate artefact format (graph, grid, neighbour sets, provenance).

## Hawkes intensity (discrete time)

A parametric discrete-time Hawkes-style model uses an intensity (conditional mean) per cell:

$$
\lambda_{j,t} = \mu_j + \sum_{k \in \mathcal{N}(j)} W(d_{jk}) \sum_{\ell=1}^{L} g(\ell)\, y_{k,t-\ell}
$$

where:

- $\mu_j \ge 0$ is a baseline rate
- $g(\ell)$ is a lag kernel over discrete lags $\ell=1..L$ (e.g. exponential)
- $W(d_{jk}) \ge 0$ downweights excitation by road travel-time distance

**TODO (M3):** wire $W(d_{jk})$ from the substrate into the core likelihood and prediction code.

## Observation model

Some workflows use an observation model (e.g. detection + clutter) to map latent counts $y^{\text{true}}$ to observed counts $y^{\text{obs}}$.

**TODO (M4):** document the observation model and its inference approximations.

## Forecasting + evaluation

The spec requires forecasting horizons of 1/3/7 days, using rolling-origin backtests and proper scoring rules.

**TODO (M5):** implement the evaluation harness (rolling backtests, CRPS/log score, calibration diagnostics).
