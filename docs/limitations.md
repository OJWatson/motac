# Assumptions and limitations

This page summarizes the modelling assumptions that make `motac` operationally useful, and the
current scope limits of the v1 implementation. The goal is not to provide internal notes, but to make
it clear what the model can be trusted to do, where caution is needed, and how to choose workflows
that remain robust in practice.

## Core assumptions behind the current formulation

The package assumes that event data can be represented as a marked count tensor with canonical shape
`[T, J, M]`: time bins (`T`), spatial nodes (`J`), and event marks (`M`). Time is discretized into fixed-width
bins (daily by default), and the forecast target is therefore short-horizon **count intensity**, not exact
continuous-time event timing.

Spatially, v1 expects either regular-grid indexing (`H*W`) or explicit graph indexing through an edge
list. Mark interactions are represented by a nonnegative matrix of the form

\[
G = \operatorname{diag}(\alpha) P,
\]

which separates per-mark excitation magnitude (`alpha`) from row-stochastic mark-routing structure (`P`).
This decomposition improves interpretability and numerical stability, but it also encodes a specific
factorization choice rather than the most general unconstrained interaction model.

## Current implementation limits in v1

Version 1 intentionally prioritizes a stable and auditable feature set. Two backend/likelihood extensions
remain deferred: the BCOO sparse backend is not yet a production path, and zero-inflated NB2 is not yet
implemented. In operational terms, this means that supported spatial operators are currently convolution
stencils and edge-list message passing, while supported observation families are Poisson and NB2.

Dataset ingestion is similarly pragmatic. Live-fetch adapters depend on optional third-party clients,
credentials, and external service availability. For reproducible research workflows, cached snapshots are
therefore preferred over purely live pulls. ACLED ingestion in particular should be treated as
cache/snapshot-first and used with explicit attention to redistribution constraints.

Finally, very large stress tests are not run by default in routine test passes. For example, the 100x100
smoke test is intentionally gated behind `RUN_SLOW_SMOKE=1`, so default CI and local quick checks remain
fast and predictable.

## Practical usage guidance under these limits

For most forecasting pipelines, `map_ensemble` is the best first-line method: it is deterministic enough
for frequent retraining and usually provides strong runtime-to-quality trade-offs. Use `svi` when interval
quality and posterior smoothness are central, and reserve `nuts_small` for tiny correctness checks or
method sanity tests rather than production-scale rolling evaluation.

In reporting, pair aggregate metrics with spatial diagnostics and calibration checks. Many apparent gains
in aggregate score can hide local hotspot failures, especially under distribution shift. The package is
designed to surface those trade-offs, but users should treat this limitations list as part of the modeling
contract when interpreting results.
