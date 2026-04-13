# Simulator: generative dynamics and controlled experiments

The simulator is the generative counterpart of the fitted mobility-Hawkes model. It is not an auxiliary
utility added for convenience; it is a first-class component for scientific debugging and stress testing.
Because simulation, fitting, and forecasting share the same recurrence logic, synthetic experiments can
be used to validate implementation semantics before expensive real-data sweeps.

In practical terms, this means the simulator supports a full closed loop:

1. draw data from known parameters,
2. fit under controlled assumptions,
3. evaluate recovery, calibration, and spatial behavior under rolling backtests.

That loop is often the fastest way to detect indexing bugs, backend mismatches, or hidden gain shifts in
temporal/spatial kernels.

## 1) Contract and inputs

`simulate_counts(...)` returns a `CountsTensor` with canonical shape `[T, J, M]`, where `T` is time bins,
`J` spatial nodes, and `M` marks. The required inputs mirror model structure:

- horizon and spatial support (`T`, `grid`, `num_nodes`, `marks`),
- connectivity operator (`ConvStencil` or `EdgeList`),
- process parameters (`HawkesSimParams`),
- optional perturbation model (`HawkesSimNoise`).

Because the output contract matches model-fitting input, synthetic datasets can be passed directly into
`MobilityHawkesModel.fit(...)` and subsequent forecast/evaluation APIs.

## 2) Core recurrence used in simulation

For each mark-basis channel, the simulator evolves latent memory by geometric decay:

$$
h_{t,j,m,b} = \rho_{m,b} h_{t-1,j,m,b} + Y_{t-1,j,m}.
$$

Spatial propagation is then applied in latent history space, mixed over temporal bases, and mapped
through mark interaction matrix

$$
G = \operatorname{diag}(\alpha)P.
$$

Conditional mean counts are

$$
\lambda_{t,j,m} = \mu_{j,m} + \sum_{m'} G_{m',m}\,z_{t,j,m'},
$$

followed by observation sampling (Poisson or NB2). This is a discrete-time analogue of marked
self-exciting dynamics in the Hawkes family {cite}`hawkes1971,ogata1988,bacry2015`.

Two implementation details are worth making explicit:

1. simulator temporal basis mixing uses provided `w_time` directly (no automatic `(1-rho)` gain factor),
2. stencil normalization is controlled by `ConvStencil.normalise` on the connectivity object itself.

Those choices should be kept in mind when designing simulation-vs-fit sensitivity studies.

## 3) Observation layers and overdispersion controls

The simulator supports two count families:

- `obs="poisson"` for equidispersed baselines,
- `obs="nb2"` for overdispersed regimes with concentration parameter.

NB2 settings can materially alter forecast sharpness and empirical coverage. For this reason, synthetic
studies that compare inference methods should report observation family and concentration assumptions
alongside fit diagnostics.

## 4) Post-simulation perturbation model

`HawkesSimNoise` applies integer-safe perturbations after trajectory generation:

- `thinning_p`: under-reporting via binomial thinning,
- `jitter_time_p`: local temporal displacement,
- `jitter_space_p`: local spatial displacement (grid neighborhood moves).

These mechanisms let users test robustness against realistic annotation and sensing artifacts while
preserving count tensor contracts expected by the rest of the stack.

## 5) Why simulator-model alignment is a reliability feature

When simulation and likelihood use the same recurrence semantics, disagreements between known truth
and fitted behavior are easier to interpret. If recovery fails badly on controlled synthetic regimes, it is a
strong signal to inspect parameterization, optimization budget, or spatial operator assumptions before
deploying on real data.

This alignment also supports principled evaluation of scoring-rule behavior and calibration drift under
known data-generating conditions {cite}`gneiting2007`.

## 6) Event-style reconstruction for diagnostics

`counts_to_events(...)` expands binned counts into event-like rows for plotting or qualitative inspection.
It is useful for visual diagnostics, but should be treated as a derived inspection view rather than a
replacement for original event-level source data.

## 7) Recommended synthetic experiment families

The simulator is especially useful for four experiment families:

- backend consistency checks (conv vs edge under matched settings),
- noise robustness sweeps (thinning/jitter stress),
- mark-coupling sensitivity (varying `P` and `alpha`),
- memory-scale sweeps (varying `rho` and temporal basis count).

For reproducible reporting, always store the random seed, connectivity configuration, observation family,
and fit/forecast settings together with results.

