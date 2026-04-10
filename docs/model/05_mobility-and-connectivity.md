# Mobility and connectivity: how spatial constraints enter

This page describes the spatial mechanism in the model from an applied modelling perspective. The key
idea is simple: contagion-like pressure should not diffuse according to raw Euclidean distance unless
that distance is a reasonable proxy for movement. In many operational settings, it is not. Movement is
channelled by roads, barriers, checkpoints, and corridors, so spatial interaction must be represented
through an explicit mobility structure {cite}`zipf1946,simini2012`.

Formally, each time step applies a spatial operator

\[
\mathcal{S}: \mathbb{R}^{J\times C} \to \mathbb{R}^{J\times C},
\]

where \(J\) is the number of nodes and \(C\) is the number of latent channels (mark-basis combinations).
The operator redistributes latent excitation before mark mixing and observation sampling. That ordering
matters: mobility determines where historical pressure can travel, and only then do cross-mark effects
amplify or damp the redistributed signal.

Unlike fully implicit spatial deep components, `motac` keeps \(\mathcal{S}\) explicit and auditable. This
is deliberate. Analysts can inspect the exact connectivity object used in simulation, fitting, and
forecasting, which keeps policy-relevant assumptions transparent.

## `ConvStencil`: dense local propagation on regular grids

The `ConvStencil` backend represents local movement as depthwise 2D convolution over regular grids. A
kernel is constructed with `make_grid_stencil(kind, size, sigma, normalise)`, where `kind` controls the
shape of local decay (`gaussian`, `manhattan`, or `moore`) and `normalise` controls whether propagation
is mass-preserving.

When normalization is enabled, stencil application is interpretable as local averaging of latent
excitation mass. When disabled, the same kernel shape acts as a gain-weighted propagator, which can be
useful for stress-testing sensitivity to spatial amplification.

In practice, this backend is usually the best baseline for dense, regular grids and high-throughput
experiments, because XLA convolution kernels are heavily optimized on both CPU and GPU.

## `EdgeList`: sparse message passing for structured mobility

The `EdgeList` backend represents movement as weighted graph transport. The constructor
`make_grid_edgelist(grid, neighbourhood, weight, directed, normalise_rows)` builds a sparse adjacency
with optional directional asymmetry and row normalization. The runtime operator `apply_edgelist`
performs weighted source-to-destination aggregation via segment sums.

This representation is useful whenever geography is not well approximated by isotropic local diffusion:
corridors, barriers, one-way connectivity, and irregular administrative networks can all be expressed as
explicit edges and weights.

## Backend dispatch and semantic consistency

`HawkesModelSpec.backend` controls dispatch. With `"auto"`, dispatch is inferred from the concrete
connectivity object on the `CountsTensor`; `"conv"` and `"edges"` enforce explicit choices; `"bcoo"` is
deferred in v1.

The same dispatcher is used by likelihood evaluation, simulation, and forecast rollout. This shared path
is an important implementation guardrail: it prevents subtle train/forecast mismatches where spatial
normalization or backend logic differs across phases. Explicit type checks also fail early if the backend
string and connectivity object disagree.

`HawkesModelSpec.normalise_spatial_kernel` can override stencil normalization at application time. This
supports controlled experiments where analysts hold kernel shape fixed while toggling between
mass-preserving and gain-amplifying propagation, without rebuilding the underlying connectivity object.

## Practical modelling guidance

A robust workflow is to start with `ConvStencil` as a reproducible grid baseline, then move to `EdgeList`
only when domain structure clearly justifies non-local or directional mobility assumptions. Whatever
backend is used, normalization choices should be kept consistent between simulation and fitting so that
interpretation of excitation mass remains stable across diagnostics and forecasting.

Future extensions can add BCOO sparse kernels and externally-derived travel-time graphs while preserving
the current high-level API and diagnostic semantics.

