```pm-status
milestone: M9
state: running
headSha: e8bfbe39f7efcd70dd33a63d8dcb93e98f63666d
ciRunUrl: https://github.com/OJWatson/motac/actions?query=branch%3Amain
updatedAtUtc: 2026-02-12T13:07:35Z
```

## Status
- M4 is **complete**:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted (structure-first):
    - `motac.loaders`, `motac.substrate`, `motac.model`, `motac.sim`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

- M5 is **complete**:
  - Minimal fit→forecast→score backtest helper exists (`motac.eval.backtest.backtest_fit_forecast_nll`).
  - Toy unit test covers basic shape/consistency expectations.

- M6 is **complete**:
  - Chicago raw loader exists (CSV + optional mobility `.npy`) and is deterministic (order preserved).
  - Minimal config + CLI entry point exists (`motac data chicago-load`).
  - Fixture-driven sanity tests cover happy path + basic validation (shape mismatch rejection).

- M7 is **complete**:
  - ACLED loader exists (CSV) and is deterministic (order preserved).
  - Minimal config + CLI entry point exists (`motac data acled-load`).
  - Fixture-driven sanity tests cover happy path + basic validation.

- M8 is **complete**:
  - CI-safe observed-data end-to-end unit test covers fit → predictive sample → score on toy data.

## Next step (M9)
- Implement and test exact observed log-likelihood v1 (start with toy-data parity checks vs current approximate workflow).

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
