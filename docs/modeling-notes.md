# Modeling notes

## Canonical shape

All core count tensors use shape `[T, J, M]`:

- `T`: time bins (typically daily)
- `J`: spatial nodes (`H*W` for grids)
- `M`: marks

## Recurrence

The latent excitation basis state follows:

`h_t = rho * h_{t-1} + y_{t-1}`

with `rho in (0, 1)` and optional multi-basis temporal mixture.

## Backends

- `ConvStencil` (default): XLA convolution over `[H, W, C]` layout.
- `EdgeList`: sparse message passing via `segment_sum`.
- BCOO is intentionally deferred from v1.

## Observation families in v1

- `Poisson`
- `NegativeBinomial2`

ZINB2 is deferred to post-v1.
