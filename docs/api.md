# API reference (concise)

## Core model workflow

- `motac.HawkesModelSpec`
- `motac.MobilityHawkesModel`
  - `.fit(data, method="map_ensemble"|"svi"|"nuts_small")`
  - `.forecast(data, fit, horizon=7, num_samples=...)`

## Data containers

- `motac.GridSpec`
- `motac.EventTable`
- `motac.CountsTensor`
- `motac.TrainTestSplit`

## Connectivity

- `motac.ConvStencil`
- `motac.EdgeList`
- `motac.make_grid_stencil(...)`
- `motac.make_grid_edgelist(...)`
- `motac.apply_conv_stencil(...)`
- `motac.apply_edgelist(...)`
- `motac.apply_spatial_backend(...)`

## Simulation

- `motac.HawkesSimParams`
- `motac.HawkesSimNoise`
- `motac.simulate_counts(...)`
- `motac.counts_to_events(...)`

## Inference and outputs

- `motac.FitConfig`
- `motac.FitResult`
- `motac.ForecastResult`

## Evaluation and backtesting

- `motac.rolling_backtest(...)`
- `motac.score_predictive_log_prob_samples(...)`
- `motac.score_log_likelihood_nb2(...)`
- `motac.score_crps_counts(...)`
- `motac.coverage(...)`
- `motac.hotspot_recall(...)`
- `motac.aggregate_daily_totals(...)`
- `motac.aggregate_mark_totals(...)`

`score_log_likelihood_nb2(...)` is kept as a backward-compatible alias for
`score_predictive_log_prob_samples(...)`.

## Datasets

- `motac.fetch_chicago_crimes(...)`
- `motac.fetch_acled_gaza(...)`
- `motac.chicago_to_counts(...)`
- `motac.acled_to_counts(...)`

## Plotting

- `motac.plot.plot_intensity_map(...)`
- `motac.plot.plot_forecast_calibration(...)`
- `motac.plot.plot_hotspots(...)`

## Benchmarks

- `motac.benchmark_backend(...)`
- `motac.compare_backends(...)`
- CLI: `motac-bench`
