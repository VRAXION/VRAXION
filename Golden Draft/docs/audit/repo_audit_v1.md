# Repo Audit v1

Date: 2026-02-05

This is a lightweight, audit-first snapshot of what's in the repo and what's missing for a "new engineer can onboard fast" experience. It is deliberately descriptive (truth visible) and avoids behavior changes.

## What exists today

### Code layout
- `Golden Code/`: "closer to production" Python package code (e.g., `Golden Code/vraxion/...`).
- `Golden Draft/`: experiments + tooling. This is where the GPU contracts and probe tooling live.
- `docs/`: GitHub Pages site (public-facing project documentation + diagrams).

### GPU measurement + reproducibility tooling
- GPU objective/stability contract: `Golden Draft/docs/gpu/objective_contract_v1.md` (doc-first; treated as a hard contract).
- Env capture tool (VRA-29): `Golden Draft/tools/gpu_env_dump.py` (writes `env.json`).
- Probe harness (VRA-32): `Golden Draft/tools/gpu_capacity_probe.py` (writes `metrics.json/csv`, `summary.md`, etc.).
- Workload ID tooling: `Golden Draft/tools/workload_id.py` (canonicalization + deterministic identifiers).
- Probe outputs / scratch: `bench_vault/` (should remain ignored; do not commit run artifacts).

### Tests
- Unit tests live under `Golden Draft/tests/` and run with:
  - `python -m unittest discover -s "Golden Draft/tests" -v`

### Repo meta + licensing
- License + commercial terms: `LICENSE`, `COMMERCIAL_LICENSE.md`
- Citation metadata: `CITATION.cff`

### GitHub plumbing
- PR template: `.github/pull_request_template.md`
- Public update issue template: `.github/ISSUE_TEMPLATE/public_update.md`
- (Currently missing) CI workflows under `.github/workflows/`

## What's missing (for a professional / reproducible repo)

These are workflow basics that reduce ambiguity for contributors and reviewers:
- **Quickstart docs**: how to run tests and basic tooling on a clean checkout (CPU and GPU probe).
- **Repro checklist**: what must be captured to make any performance claim verifiable (commit, seed, `env.json`, `workload_id`, args, outputs).
- **Governance files**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- **CI**: CPU-only test run on PRs so regressions get caught automatically (no CUDA required).

## Risky cleanup candidates (do not do in v1)

These are real, but fixing them tends to create large diffs or break scripts:
- Space-in-path directories (`Golden Draft/`, `Golden Code/`) make some tooling awkward on Linux/shell scripts.
- Drift between public site (`docs/`) and internal contracts/tools (`Golden Draft/docs/`).
- Run artifact sprawl (`bench_vault/`, `logs/`): must remain ignored and never committed.

## Immediate quick wins (<= 1 day, PR-safe)

This repo-professionalization v1 intentionally focuses on small PRs:
1) Audit snapshot + entrypoints map (this doc + `tree_depth6.txt` + `entrypoints_v1.md`).
2) Quickstart + reproducibility docs + small `README.md` upgrade.
3) Contribution/security templates so GitHub intake is structured.
4) CPU-only CI that runs `unittest` + `compileall` on PRs.

## Deferred changes (explicitly out of scope for v1)

- Rename directories to remove spaces and standardize casing.
- Convert to an installable Python package with a single CLI entrypoint.
- Add Ubuntu CI and/or self-hosted GPU runners.
- Consolidate experimental scripts/runners (once Chapter gates allow it).
