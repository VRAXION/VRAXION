# VRAXION Codex Handover

Last updated: 2026-06-15

## Start Here

```text
repo = VRAXION_anchorwiki
branch = main
latest_release_target = v6.1.7
current_evidence_anchor = f32a6f4b Finalize E127 cycle 40 checkpoint
current_status = E127 scoped Orange/Legendary text-operator farm, cycle 40
```

This is the first file a fresh Codex should read after cloning the repo.

## Current State

E127 completed a supervised overnight cyclic operator farm:

```text
cycles = 40
Orange/Legendary scoped operators = 382
mutation attempts = 1,849,625
accepted mutations = 12,123
rollbacks = 1,837,502

hard negatives = 0
false commits = 0
wrong scope calls = 0
unsupported answers = 0
```

The latest text-to-text smoke is a deterministic operator + template renderer,
not an LLM/freeform generation claim. It shows proto-assistant behavior over
short prompts using the governed operator library.

## Claim Boundary

Allowed:

```text
VRAXION v6 has a governed operator-library checkpoint with 382 scoped
Orange/Legendary text operators, continuous checkpointing discipline, a
dashboard sample pack, and a deterministic text-to-text render smoke.
```

Not allowed:

```text
open-domain chatbot
Gemma/GPT-level generation
production assistant
PermaCore / TrueGolden promotion
consciousness / sentience
unbounded reasoning claim
```

## Key Files

```text
README.md
docs/CURRENT_STATUS.md
docs/GETTING_STARTED.md
docs/VERSION.json
CHANGELOG.md

docs/research/E127_OVERNIGHT_TEXT_SKILL_FARM_ORANGE_CYCLE_RESULT.md
docs/research/E127_TEXT_TO_TEXT_RENDER_SMOKE_CURRENT_RESULT.md
docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle/
docs/research/artifact_samples/e127_text_to_text_render_smoke_current/
```

## Legal / License

The root [`LICENSE`](LICENSE) is the controlling license text. VRAXION uses the
custom **VRAXION Community Source License 1.0**: broad free community use, but
not OSI-approved open source.

Commercial monetization of third-party access to VRAXION-powered functionality
is Royalty Use under the license unless a separate written agreement exists.
The default royalty is 19% of Attributable Net Revenue:

```text
1% Founder Allocation
18% VRAXION Forever Prize Allocation
```

Read these before changing release/legal language:

```text
LICENSE
legal/LEGAL.md
legal/COMMERCIAL_USE_GUIDE.md
legal/VRAXION_FOREVER_PRIZE_CHARTER.md
docs/legal/PRIOR_ART_AND_PROVENANCE_CHECKLIST.md
```

## Dashboard

Generate the local operator dashboard:

```powershell
python scripts/tools/generate_operator_rank_dashboard.py --e127 docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle --out target/pilot_wave/operator_rank_dashboard/index.html
```

Expected E127 dashboard cards:

```text
E127 ORANGE = 382
E127 CYCLES = 40
E127 HARD NEGATIVES = 0
E127 FALSE COMMITS = 0
```

## Quick Checks

Run these after clone or before continuing:

```powershell
git status --short --branch
python -m py_compile scripts/probes/run_e127_overnight_text_skill_farm_orange_cycle.py scripts/tools/generate_operator_rank_dashboard.py
python -m compileall -q scripts
cargo test --workspace
git diff --check
```

If the machine is slow or missing Rust tooling, report the dependency state
honestly instead of faking results.

## Operating Rules

1. No black-box runs. Long jobs must write partial outcomes every few minutes or
   faster.
2. Do not run tiny sequential compute if a real multi-seed/multi-worker run is
   the actual goal.
3. Do not stage unrelated dirty or untracked files.
4. Push clean successful checkpoints to `origin/main`.
5. Keep claim boundaries explicit.

## Current Untracked Local Leftovers

At the time of this handover, these old sample artifact directories may exist
locally as untracked files. They were intentionally left untouched:

```text
docs/research/artifact_samples/e31_breakpoint_ladder_and_bottleneck_localization/
docs/research/artifact_samples/e32_flow_field_capacity_and_trace_ledger_repair_confirm/
docs/research/artifact_samples/e44_abstract_payload_wire_capacity_smoke/
docs/research/artifact_samples/e44b_parallel_abstract_wire_stream_field_smoke/
docs/research/artifact_samples/e44c_reserve_wire_mask_and_noise_stress/
docs/research/artifact_samples/e63_pocket_observatory_visual_debug_dashboard/
docs/research/artifact_samples/e64_field_size_and_agency_view_sweep/
docs/research/artifact_samples/e65_locked_body_runtime_integration_preflight/
```

Do not delete or commit them without an explicit cleanup decision.

## Next Work

Recommended next steps:

```text
1. Decide whether to continue E127 with a fresh candidate pack or pause farming.
2. Run a broader deterministic text-to-text smoke over more prompts.
3. Build the next bridge from operator/template rendering toward richer text IO.
4. Keep all new skills scoped, dashboard-visible, and rollback-safe.
```
