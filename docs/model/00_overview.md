# Overview

This package is concerned with **short-horizon probabilistic forecasting of spatial event data**. The
inputs are event records with a time stamp, a location (latitude/longitude), and a *mark* (event type).
The outputs are predictive distributions for the number of events that will occur in each region over
the next few days, together with uncertainty intervals that remain meaningful under repeated
evaluation.

The modelling choice in this project is to treat event logs as a form of **self-exciting spatio-temporal
stochastic process**: events are not independent, and the recent history of events contains
information about near-future risk. Self-exciting processes of this kind are classically formalised by
Hawkes processes and their marked / multivariate extensions {cite}`hawkes1971,bacry2015,rizoiu2017`.

In many applied settings, however, spatial propagation is not well described by Euclidean distance.
Movement follows roads, corridors, and barriers, and the relevant notion of distance is often
*effective travel cost* rather than geometry. The central idea of this package is therefore to build a
marked Hawkes-like forecasting model whose spatial triggering kernel is constrained by explicit
**mobility graph operators** (stencils or edge-lists), preserving interpretability of spatial interaction
assumptions while remaining computationally practical for repeated rolling-origin forecasting
{cite}`zipf1946,simini2012`.

## From events to a spatio-temporal count field

The primary data interface mirrors how forecasting is typically evaluated in practice. We begin with an
event table

$$
  e_i = (t_i, \mathrm{lat}_i, \mathrm{lon}_i, m_i), \qquad i = 1,\ldots,N,
$$

and discretise into **daily marked counts** on a set of spatial nodes (usually a grid):

$$
  y_{t,j,m} \in \{0,1,2,\ldots\}, \qquad t=1,\ldots,T,\ j=1,\ldots,J,\ m=1,\ldots,M.
$$

This discretisation is not a modelling assumption so much as an engineering choice: it makes
rolling backtests, calibration checks, and operational forecasting straightforward, while retaining
enough resolution for short-term hotspot forecasting.

In code, this pipeline is represented by two minimal containers:
`EventTable` (event-level records) and `CountsTensor` (the discretised tensor plus spatial metadata).
All downstream modelling and evaluation depends only on these objects.

## Mobility-constrained self-excitation

The models implemented in this package decompose the expected event intensity into two parts:
(i) a **baseline risk surface** (slowly varying in time and space), and
(ii) a **self-excitation term** that propagates recent activity across the mobility graph.

The resulting model can be read as a discrete-time approximation to a marked Hawkes process
{cite}`hawkes1971,bacry2015`, with the important modification that spatial triggering is mediated by
a graph distance or travel-time impedance rather than by Euclidean kernels.

## Effective mobility in v1

Even with a good travel-time graph, two links with similar impedance may correspond to different
interaction intensity (e.g. because of barriers, checkpoints, or corridor effects). In the current v1
implementation, this is handled through explicit spatial operators (`ConvStencil` or `EdgeList`) and
their weights/normalisation choices, rather than a required learned embedding block.

This design keeps the mobility assumptions auditable: analysts can inspect and version the exact
connectivity object used in simulation, fitting, and forecasting. Learnable representation components are
an extension path, but they are not required for the baseline model currently documented here.

## Inference and calibrated uncertainty

Forecast evaluation is sensitive to uncertainty calibration: we care about predictive distributions, not
only point forecasts. Exact Bayesian inference is typically infeasible for these models, so we follow the
practical strategy used in scalable Bayesian neural field models and deep-ensemble literature
{cite}`lakshminarayanan2017,gelman2013`: approximate posterior uncertainty using **ensembles** (multiple MAP
fits, or mixtures of variational posteriors). This yields coherent posterior predictive distributions and is
computationally compatible with rolling-origin backtests.

In practical terms, this means prioritising calibration-aware forecast quality metrics and predictive
distribution diagnostics over single point-forecast error summaries, in line with broader forecasting
evaluation guidance {cite}`gneiting2007`.

## Reading guide

The next three pages form the core “paper-style” description of the method:

- the model specification and its connection to marked Hawkes processes
  (`Mobility-constrained marked Hawkes model`),
- the inference algorithms (`Inference`), and
- the forecasting protocol and evaluation metrics (`Forecasting`).

The simulator and dataset pages then provide end-to-end vignettes demonstrating that the same
interface supports controlled experiments (simulator) as well as real event datasets.

## References

```{bibliography}
:cited:
```
