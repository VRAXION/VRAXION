# INSTNCT RC_001 Install Guide

Status: release candidate install guide for local/private evaluation only.

This guide describes how to prepare a clean checkout for `INSTNCT_RC_001`
validation. It is not GA, not production deployment, not hosted SaaS launch,
not public beta, and not final legal terms.

## Prerequisites

- Rust toolchain capable of building `instnct-core`.
- Python 3 for the deployment harness and static checkers.
- PowerShell for the Windows convenience smoke wrapper.
- A clean checkout of this repository on `main`.

## Local/Private Evaluation Install

Use the repository checkout directly. Do not create or commit release archives,
binaries, checkpoints, `target/`, `node_modules/`, or `.svelte-kit/` content as
part of RC_001 validation.

Recommended output location for local/private evaluation:

```text
target/pilot_wave/stable_loop_phase_lock_061_release_candidate_package/
```

The install path is intentionally local/private evaluation only. It does not
authorize external use.

No hosted SaaS launch.
No production deployment.
No public beta.
No clinical use.
No high-stakes education use.
No production API readiness.
No final legal terms.

## Static Package Check

Run the RC static checker:

```powershell
python scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py --check-only
```

The checker validates committed files only. It does not run smoke commands and
does not create a release archive, binary, checkpoint, or model experiment.

## Boundary

Exact boundary tokens:

```text
no GA
no production deployment
no hosted SaaS launch
no public beta
no production API readiness
no production readiness
no clinical use
no high-stakes education use
no final legal terms
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
