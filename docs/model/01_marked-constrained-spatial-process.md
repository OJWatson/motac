# Mobility-constrained marked spatial Hawkes process

This page gives a paper-style mathematical description of the model implemented in `motac`.
The construction follows the marked Hawkes perspective {cite}`hawkes1971,bacry2015,rizoiu2017`, but
is adapted to operational short-horizon count forecasting on mobility-constrained spatial supports.

## 1) Data and indexing

We work with binned marked counts

$$
Y_{t,j,m} \in \{0,1,2,\ldots\}, \quad t=1,\ldots,T,\; j=1,\ldots,J,\; m=1,\ldots,M.
$$

- $t$: time bin (typically day)
- $j$: spatial node (grid cell or node in a mobility graph)
- $m$: mark/event type

The canonical in-memory tensor shape is `[T, J, M]`.

## 2) Conditional intensity decomposition

For each $(t,j,m)$, define conditional mean count

$$
\lambda_{t,j,m} = \mu_{j,m} + \sum_{m'=1}^{M} G_{m',m}\; z_{t,j,m'}.
$$

where:

- $\mu_{j,m} \ge 0$: baseline risk
- $G \in \mathbb{R}_{+}^{M\times M}$: mark-to-mark triggering matrix
- $z_{t,j,m'}$: mobility-filtered latent excitation for source mark $m'$

This additive baseline-plus-excitation form is deliberately interpretable: baseline terms absorb slow
spatial heterogeneity while excitation captures near-term contagion-like spillovers.

In code, the mark matrix is parameterized as

$$
G = \operatorname{diag}(\alpha) P,
$$

with $\alpha_m \in (0, \alpha_{\max})$, and each row of $P$ simplex-constrained.

## 3) Temporal basis recurrence

For each mark $m$, we keep $B$ latent basis states per node:

$$
h_{t,j,m,b} = \rho_{m,b}\, h_{t-1,j,m,b} + Y_{t-1,j,m},
\qquad \rho_{m,b} \in (0,1).
$$

This is a geometric-decay bank of memory traces (short/long memory controlled by $\rho$).

Write flattened per-node channels as

$$
\tilde h_{t,j,:} \in \mathbb{R}^{M\cdot B}.
$$

## 4) Mobility-constrained spatial propagation

Let $\mathcal{S}(\cdot)$ be the spatial operator induced by the chosen connectivity backend:

- convolution stencil (`ConvStencil`) on regular grids, or
- sparse edge message passing (`EdgeList`) on graph neighborhoods.

Then

$$
\hat h_{t,:,:} = \mathcal{S}(\tilde h_{t,:,:}).
$$

Reshape to $[J,M,B]$, and combine basis channels with simplex weights
$w_{m,:}$:

$$
z_{t,j,m} = \sum_{b=1}^{B} w_{m,b}\; \hat h_{t,j,m,b},
\qquad w_{m,:} \in \Delta^{B-1}.
$$

So mobility enters *before* mark mixing: local/graph spread is applied in the latent history space.

## 5) Observation layer

Conditionally on $\lambda_{t,j,m}$, observations are sampled either from a Poisson law,

$$
Y_{t,j,m} \sim \operatorname{Poisson}(\lambda_{t,j,m}),
$$

or from NB2,

$$
Y_{t,j,m} \sim \operatorname{NB2}(\text{mean}=\lambda_{t,j,m},\; \text{concentration}=\phi_m),
$$

with variance $\lambda_{t,j,m} + \lambda_{t,j,m}^2/\phi_m$. In the current implementation,
$\phi_m$ is mark-specific but shared across nodes, which is a deliberate parsimony choice: it captures
systematic overdispersion differences across event types while keeping the parameter dimension stable in
large spatial grids.

The recursion is initialized with baseline-only intensity at the first bin. In other words, the model scores
$Y_1$ under $\lambda_1=\mu$, then uses $Y_1$ to update excitation for $t=2$. This is standard in
discrete latent-state Hawkes approximations and avoids introducing an extra unknown pre-history state.

## 6) Parameterization and constraints

Inference is performed in unconstrained coordinates and mapped to admissible parameters through smooth
transforms:

$$
\mu = \operatorname{softplus}(\mu_{\text{raw}})+\epsilon,\quad
\alpha = \alpha_{\max}\,\sigma(\alpha_{\text{raw}}),\quad
P = \operatorname{softmax}(P_{\text{raw}}\;\text{row-wise}),
$$

$$
\rho = \sigma(\rho_{\text{raw}}),\quad
w = \operatorname{softmax}(w_{\text{raw}}\;\text{row-wise}),\quad
\phi = \operatorname{softplus}(\phi_{\text{raw}})+\epsilon.
$$

Two model-spec normalisation switches control how latent excitation mass is interpreted. The spatial flag
`normalise_spatial_kernel` determines whether stencil kernels are normalized at application time; the
temporal flag `normalise_time_kernel` controls whether each basis contributes with scale $1-\rho_{m,b}$
so that temporal basis components integrate to unit mass. Both options are useful in sensitivity analysis:
one can compare "mass-preserving" versus "free-gain" dynamics without changing core code paths.

## 7) Full conditional recursion

The computational graph at each time step is

$$
(h_{t-1},Y_{t-1}) \mapsto h_t \mapsto \hat h_t = \mathcal{S}(h_t)
\mapsto z_t \mapsto \lambda_t \mapsto \log p(Y_t\mid\lambda_t).
$$

This pipeline is implemented using `jax.lax.scan` (and an equivalent NumPyro scan path for Bayesian
inference), so the recurrence is expressed once and executed as an XLA-compiled loop over time. The
main practical implication is that the model remains computationally predictable when repeatedly re-fit
across rolling origins.

## 8) Relation to classical Hawkes models

Relative to continuous-time marked Hawkes models {cite}`hawkes1971,bacry2015`, this construction can
be viewed as a discretized, mobility-constrained branching mechanism. The branching interpretation is
retained at the mark level through $G$, while temporal and spatial propagation are represented by
learned basis recursions and explicit connectivity operators. The resulting object is therefore still a
self-exciting process model, but optimized for operational short-horizon count forecasting where data are
already consumed in daily bins.

## 9) Stability, subcriticality, and identifiability

In multivariate Hawkes theory, stability is commonly tied to a spectral-radius condition on effective
branching gain. In this model the relevant matrix is not only $G$, but an effective matrix that also
depends on temporal memory mass (via $\rho,w$). Consequently, bounding each $\alpha_m$ below one is
necessary but not sufficient for strict subcriticality in all parameter regimes.

For this reason, fit diagnostics now report posterior/sample summaries of the effective excitation spectral
radius (mean, max, and subcritical fraction). The package does not hard-project parameters onto the
subcritical region, because that can over-constrain practical forecasting fits, but it exposes enough
diagnostic structure to make potential instability visible in evaluation workflows.

Identifiability remains partially shared between baseline and excitation pathways, especially in low-count
regimes. The constrained parameterization, weakly informative priors, and repeated-origin validation are
used together to keep this trade-off manageable in practice.

## 10) Why this formulation is operationally robust

The model is intentionally designed around static tensor contracts, differentiable transforms, and shared
recurrence code across simulation, MAP, SVI, and forecasting. This alignment reduces implementation
drift, simplifies testing, and makes model behavior easier to audit. In short, the formulation aims to
preserve Hawkes-style interpretability while remaining robust under repeated refit-and-forecast loops in
real deployment-like backtesting.

