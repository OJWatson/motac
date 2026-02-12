```pm-status
milestone: M5
state: running
headSha: edfb3f190e8b063eb93b48c7905b3787936b46a7
ciRunUrl: https://github.com/OJWatson/motac/actions/runs/21936278420
updatedAtUtc: 2026-02-12T06:58:52Z
```

## Status
- M4 is **complete**:
  - Spec checklist + mapping: `docs/spec_alignment.md`
  - Architecture summary: `docs/architecture.md`
  - Package boundaries extracted (structure-first):
    - `motac.loaders`, `motac.substrate`, `motac.model`, `motac.sim`, `motac.eval`, `motac.inference`, `motac.cli`, `motac.configs`
  - CLI internal boundary extracted: `motac.cli.commands` (while keeping the entry point stable)
  - Import-path smoke tests added for new packages.

- M5 is **in progress**.

## Next step (M5)
- Land an end-to-end toy test: fit -> forecast -> score (mean NLL) using the road-constrained parametric model.
- Add a Negative Binomial mean NLL smoke test (dispersion provided) to exercise the metric API.

## Notes
- Documentation builds are tracked in CI; the status header points at the last meaningful code gate.
- Keep commits small and CI-gated; update this file on each push/gate.
