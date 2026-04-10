# Forecasting and posterior predictive rollout

This page explains how `motac` produces future trajectories once a model has been fitted. The forecast is
not treated as deterministic extrapolation from a point estimate. Instead, it is constructed as a
posterior predictive simulation problem, so both parameter uncertainty and stochastic count variation are
carried into horizon-level risk summaries {cite}`gelman2013`.

## Forecast target and Monte Carlo approximation

Let \(Y_{1:T}\) denote observed history, \(H\) the forecast horizon, and \(\theta^{(s)}\) parameter draws from
an approximate posterior. Forecasting draws trajectories according to

\[
Y_{T+1:T+H}^{(s)} \sim p(\cdot \mid Y_{1:T}, \theta^{(s)}),
\]

which is a Monte Carlo approximation to the full posterior predictive integral

\[
p(Y_{T+1:T+H}\mid Y_{1:T}) = \int p(Y_{T+1:T+H}\mid Y_{1:T},\theta)\,p(\theta\mid Y_{1:T})\,d\theta.
\]

The practical consequence is that forecast uncertainty is represented directly in samples, rather than
imposed later through ad hoc error bars.

## Conditioning on the correct terminal latent state

A subtle but crucial step is state initialization. Forecast rollout first replays the training history
through the same recurrence used during fitting to recover the terminal latent memory state \(h_T\). In
other words, forecasting starts from the inferred excitation state implied by observed history, not from a
zero or arbitrary warm start.

Implementation order is aligned with the likelihood path: the first observation is scored under baseline,
then contributes to downstream excitation. During forecast rollout, each simulated count draw is fed back
into the recurrence before generating the next step. This avoids the common off-by-one mismatch where fit
and forecast semantics silently diverge.

## One-step transition and autoregressive uncertainty propagation

For each horizon step, the model updates temporal basis states, applies the spatial mobility operator,
mixes basis contributions with \(w\), applies mark coupling via \(G\), adds baseline \(\mu\), and samples from
the observation law (Poisson or NB2). Repeating this transition yields fully autoregressive trajectories.

Because each step depends on realized simulated counts from previous steps, uncertainty naturally expands
with horizon when dynamics are noisy or strongly self-exciting. This behavior is desirable in operational
forecasting: confidence should degrade when the process itself is uncertain.

Importantly, normalization settings are preserved between fitting and forecasting. If
`normalise_time_kernel=True`, basis contributions are scaled by \((1-\rho_{m,b})\); if
`normalise_spatial_kernel` is enabled, stencil normalization is applied (or overridden) during spatial
dispatch. Keeping these semantics identical across phases avoids hidden gain shifts at deployment time.

## Predictive outputs and calibration-relevant summaries

`ForecastResult` returns sampled trajectories `y_samples` with shape `[S, H, J, M]`, together with Monte
Carlo means and configurable quantiles (default 0.05, 0.5, 0.95). Because these summaries are computed
directly from trajectory samples, interval width reflects both epistemic uncertainty (across parameter
draws) and aleatoric uncertainty (observation noise).

When intervals are persistently under- or over-dispersed in backtests, this usually signals model-choice
or inference-choice issues rather than a plotting artifact. In practice, users should interpret coverage
diagnostics as feedback on the full fit-and-forecast system.

## Rolling-origin evaluation as the primary protocol

The package’s `rolling_backtest` routine repeatedly fits on a temporal prefix, forecasts the next horizon,
scores predictions, and advances by a fixed step. This protocol is preferred over random splitting for
temporally dependent event data because it respects causal ordering and emulates real deployment
conditions.

Evaluation combines several complementary metrics: smoothed predictive log-score, CRPS-style count score,
interval coverage, and hotspot recall. Together these characterize sharpness, reliability, and spatial
targeting quality, consistent with proper-scoring-rule forecasting practice {cite}`gneiting2007`.

## Practical interpretation and common failure patterns

In applied use, stable quantile estimation requires enough predictive draws, and aggregate metrics should
always be paired with origin-level traces. It is common for aggregate performance to look acceptable while
specific origins or hotspot-localization behavior degrade. For this reason, decision-oriented reporting
should include both global summaries and split-level diagnostics.

