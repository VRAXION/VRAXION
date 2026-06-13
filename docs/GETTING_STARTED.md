# Getting Started with VRAXION

_Last updated: 2026-05-19_

## Current Status

```text
Local/private bounded AI stack: release-ready
GPT-like / open-domain assistant: not ready
Production/public service: not claimed
```

Start with [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) for the current status and claim boundary.

## Repository Layout

| Path | Purpose |
|---|---|
| `instnct-core/` | Rust architecture research surface |
| `tools/instnct_service_alpha/` | Localhost/private bounded chat API alpha |
| `tools/instnct_deploy/` | Local/private deployment harness |
| `scripts/probes/` | Retained 078-100 proof/eval runners and checkers |
| `docs/research/` | Contract/result evidence for the research runway |
| `docs/wiki/` | Repo-tracked source mirror for the GitHub wiki |
| `Python/` | Historical/reference deploy SDK blocks |
| `target/` | Generated local evidence artifacts; ignored by Git |

## Claim Boundary

Allowed wording:

> VRAXION has a hash-checked, audited, localhost/private bounded-domain AI stack that passed artifact, runtime, API, harness, OOD/red-team, long-run, package, generation-repair, and clean local/private deploy-readiness gates.

Do not claim:

- production-ready public AI
- public API
- hosted SaaS
- GPT-like assistant readiness
- open-domain chat readiness
- safety-aligned production system
- proof that INSTNCT/AnchorRoute is an open-domain LM winner

## Quick Verification

If the local generated `target/` evidence is present, run:

```powershell
python scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale_check.py --check-only
```

For source-level sanity:

```powershell
python -m py_compile scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale.py
python -m compileall Python tools
cargo test -p instnct-core
```

Generated packages/checkpoints are not committed. If a checker reports missing `target/` artifacts in a fresh clone, regenerate the relevant smoke run or obtain the private evaluation bundle.

## Main Evidence Chain

Bounded local/private release-ready:

```text
083 -> 084 -> 085 -> 086 -> 087 -> 088 -> 089 -> 089B -> 096 -> 097 -> 098 -> 099
```

Open-vocab assistant capability research:

```text
091 -> 092 -> 093 -> 094 -> 094B -> 095 -> 096 -> 097 -> 100
```

## Reading Order

| Step | Document | Why |
|---|---|---|
| 1 | [`README.md`](../README.md) | Front-door summary and claim boundary |
| 2 | [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current official state |
| 3 | [`docs/research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RESULT.md`](research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RESULT.md) | Bounded release-ready result |
| 4 | [`docs/research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_RESULT.md`](research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_RESULT.md) | Current capability-scale research result |
| 5 | [`VALIDATED_FINDINGS.md`](../VALIDATED_FINDINGS.md) | Historical evidence summary |
| 6 | [`docs/REPO_CONSOLIDATION_2026_05_19.md`](REPO_CONSOLIDATION_2026_05_19.md) | What was archived/removed from active main |

## Archive Branch

The pre-cleanup source tree is preserved at:

```text
archive/pre-consolidation-20260519-main-snapshot
```

Use it for old scratch tools and non-current probe runners that were removed from active `main`.

## License

- Community source license: [`LICENSE`](../LICENSE)
- Commercial royalty terms and brand rights: [`legal/LEGAL.md`](../legal/LEGAL.md)
- VRAXION Forever Prize charter: [`legal/VRAXION_FOREVER_PRIZE_CHARTER.md`](../legal/VRAXION_FOREVER_PRIZE_CHARTER.md)
- Citation: [`CITATION.cff`](../CITATION.cff)
