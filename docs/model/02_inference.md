# Inference: MAP ensembles, SVI, and small NUTS

Inference in `motac` is designed around a practical requirement: the model must be re-fit repeatedly
under rolling-origin evaluation without becoming numerically fragile or operationally too expensive. For
that reason, the package emphasizes approximate Bayesian workflows that preserve uncertainty information
while remaining stable in day-to-day forecasting pipelines.

At a high level, all methods target the same posterior quantity

\[
\log p(\theta \mid Y) = \log p(Y\mid\theta) + \log p(\theta) + C,
\]

with \(\log p(Y\mid\theta)\) computed by the scan-based model recurrence and \(\log p(\theta)\) defined on
unconstrained parameter coordinates. This follows the usual Bayesian decomposition {cite}`gelman2013`,
but the implementation focus is explicitly operational: fast repeated fitting with transparent diagnostics.

## Prior structure in unconstrained coordinates

Priors are placed on raw parameters before positivity/simplex transforms, which makes optimization and
variational learning more stable than directly constraining variables in-place. In the current
implementation, the baseline raw scale is centered near low daily intensity (`mu_raw ~ N(-3, 1)`), while
interaction and memory raw terms are weakly regularized around zero. The dispersion raw term has a prior
centered at moderate over-dispersion (`phi_raw ~ N(3, 0.5)`).

After transformation, these become weakly informative priors over admissible model parameters
\((\mu, \alpha, P, \rho, w, \phi)\), preserving interpretability while discouraging unstable extremes early
in optimization.

## MAP ensembles as the default operational posterior approximation

The default method, `map_ensemble`, optimizes the posterior objective from multiple random initial
conditions. Each restart is trained with gradient clipping and early stopping, and only the best state per
restart is retained. Collecting these retained optima gives a practical ensemble approximation to
epistemic uncertainty:

\[
\{\hat\theta^{(s)}\}_{s=1}^S.
\]

This strategy intentionally mirrors the motivation behind deep ensembles {cite}`lakshminarayanan2017`:
exact posterior sampling is often too expensive for repeated short-horizon backtests, but multiple
independent high-quality optima capture meaningful uncertainty variation at low additional implementation
complexity.

In `FitResult`, ensemble members are exposed through `posterior_samples`, so forecast sampling can treat
MAP ensembles and Bayesian methods through the same downstream API.

## Stochastic variational inference (SVI)

When a smoother posterior family is needed, `svi` uses NumPyro autoguides (`AutoDiagonalNormal` or
`AutoNormal`) and optimizes an ELBO objective

\[
\mathcal L(q) = \mathbb E_{q(\theta)}[\log p(Y,\theta)-\log q(\theta)].
\]

The implementation prefers `AutoContinuousELBO` when available and falls back to `Trace_ELBO` for
compatibility. Posterior draws from the guide are transformed back to constrained model space and exposed
in the same `FitResult.posterior_samples` contract used by MAP ensembles.

In practice, SVI is useful when one wants a more explicitly parameterized approximate posterior and can
afford additional optimization time.

## Small NUTS as a calibration and debugging bridge

`nuts_small` is intentionally scoped to tiny problems. It is not the production pathway for large-scale
rolling forecasting, but it is important for model checking: if MAP/SVI behavior diverges sharply from
small MCMC runs on toy instances, that often indicates a parameterization or optimization issue worth
investigating.

## Unified output contract and diagnostics

All inference methods return a `FitResult` with method label, constrained posterior samples, and
diagnostics. This method-invariant contract is central to the package architecture: downstream forecasting
and evaluation code should not need to care whether uncertainty came from MAP restarts, variational
samples, or small MCMC draws.

Diagnostics are designed for operational monitoring. Beyond objective/loss traces, the package reports
effective-excitation spectral-radius summaries (mean, max, subcritical fraction), which are particularly
useful for spotting near-instability regimes before they surface as forecast pathologies.

## Practical interpretation

MAP ensembles and diagonal variational families are approximations, not exact posterior representations.
They are used because they provide a useful calibration-versus-runtime balance in repeated forecasting
workflows. In this project, that balance is the key criterion: uncertainty quality must be good enough to
support calibrated decisions, but fitting must remain fast and robust across many rolling-origin refits.

