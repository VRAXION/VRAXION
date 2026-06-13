# VRAXION

VRAXION is building **INSTNCT**: a gradient-free self-wiring architecture that learns by changing its own directed graph instead of running backpropagation through a fixed topology.

> **Core thesis**
>
> **Inference emerges as the fixed point of destructive interference.**
>
> Signal enters a structured recurrent substrate, incompatible propagation paths cancel through destructive interference, and the surviving pattern — the fixed point — is read out as inference. This is a research thesis for the architecture line, not a separate claim of achieved sentience.

This repository is meant to be a credible front door for technical buyers and engineers. It should let a first-time reader answer five things quickly:

1. what VRAXION is,
2. why the architecture is different,
3. what is actually proven,
4. what the current canonical code path is,
5. how to verify one claim in minutes.

## Release Snapshot

- **Current repo metadata release:** `v5.0.0-beta.33` in [`docs/VERSION.json`](docs/VERSION.json).
- **Latest GitHub release with public release assets:** [`v5.0.0-beta.8`](https://github.com/VRAXION/VRAXION/releases/tag/v5.0.0-beta.8). Later beta heartbeat commits are source/evidence updates, not broad public-production releases.
- **Bounded local/private stop condition:** **positive** as of `099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE`. This means the bounded-domain local/private stack is release-ready for private/local evaluation under the recorded claim boundary.
- **Capability track:** `100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE` is positive, but remains research-only. It is not GPT-like readiness, not public API readiness, not production chat, and not an open-domain assistant claim.
- **Current bounded service path:** local/private bounded chat stack through `tools/instnct_service_alpha/` and `tools/instnct_deploy/`, with evidence in the 083-100 probe lineage.
- **Current architecture research path:** [`instnct-core/examples/neuron_grower.rs`](instnct-core/examples/neuron_grower.rs) remains the Rust architecture line; open-vocab assistant-scale work is runner-local research, not a replacement architecture claim.
- **Python deploy SDK:** [`Python/`](Python/) — Block A + B, pure numpy, no ML framework dependency
- **Historical Python research lane:** frozen at tag [`archives/python-research-20260420`](https://github.com/VRAXION/VRAXION/tree/archives/python-research-20260420) (was `instnct/` — archived 2026-04-20 after migration to Rust `instnct-core/`)
- **Pre-consolidation source archive:** [`archive/pre-consolidation-20260519-main-snapshot`](https://github.com/VRAXION/VRAXION/tree/archive/pre-consolidation-20260519-main-snapshot) preserves the full tree before the May 19 cleanup.

## Current Official Status

```text
Local/private bounded AI stack: release-ready under local/private claim boundary
GPT-like / open-domain assistant: not ready
Production/public service: not claimed
```

The validated bounded stack is:

```text
083 model artifact RC
-> 084 local inference runtime
-> 085 localhost/private bounded chat API alpha
-> 086 deployment harness integration
-> 087 OOD/red-team service eval
-> 088 long-run/concurrency stability
-> 089 private evaluation RC package
-> 089B packaged winner reproducibility proof
-> 096 fresh chat generation eval
-> 097 multi-seed decoder/OOD/retention confirm
-> 098 private evaluation RC refresh
-> 099 clean local/private deploy-ready gate
```

The current research-only capability track is:

```text
091 open-vocab byte-LM foundation
-> 092 FineWeb slice confirm
-> 093 FineWeb margin/scale confirm
-> 094 chat SFT mix PoC
-> 094B free-generation gap analysis
-> 095 decoder generation repair
-> 096 fresh generation eval
-> 097 multi-seed OOD/retention confirm
-> 100 assistant capability scale probe
```

The safe external wording is:

> VRAXION has a hash-checked, audited, localhost/private bounded-domain AI stack that passed artifact, runtime, API, harness, red-team, long-run, private package, generation-repair, and clean local deploy-readiness gates. Separately, runner-local open-vocab assistant capability probes show improvement, but do not prove GPT-like readiness.

Forbidden overclaims:

- production-ready public AI service
- public API or hosted SaaS
- GPT-like assistant readiness
- open-domain chat readiness
- safety-aligned production system
- proof that INSTNCT/AnchorRoute is an open-domain LM winner

## Why This Architecture Is Different

INSTNCT is built around a small set of unusual choices:

- **Stagewise self-wiring**: the graph grows neuron by neuron instead of optimizing a fixed dense topology.
- **Scout-first search**: a cheap all-signal probe ranks promising parents and pair interactions before exhaustive ternary search.
- **Bias-free threshold neurons**: the persistent grower now stores neurons directly as `dot >= threshold`, without redundant bias search.
- **Append-only evidence**: canonical runs are expected to emit reproducible evidence bundles and resumable state, not just ad hoc logs.

The canonical grower contract is [`docs/GROWER_RUN_CONTRACT.md`](docs/GROWER_RUN_CONTRACT.md).

## Status Taxonomy

To keep the public story truthful, this repo uses three labels consistently:

- **Current mainline**: what is actually shipped in code on `main`.
- **Validated finding**: a result backed by a concrete experiment, but not yet promoted into the canonical code path.
- **Experimental branch**: an active build target or design direction that is not yet a validated default.

If code and docs disagree, **code wins for “Current mainline.”**

The repo-tracked docs are the canonical public source. The GitHub wiki is a secondary mirror, not an independent source of truth.

Historical archives are preserved as immutable tags (no long-lived archive branches remain). See [`ARCHIVE.md`](ARCHIVE.md) for the full list, including the previous-era surface freezes and per-cleanup branch-head snapshots.

Retired line names and older local folders belong in [Project Timeline](https://github.com/VRAXION/VRAXION/wiki/Timeline-Archive), not in the current front-door stack.

## Current State

### Current mainline

- The local/private bounded chat stack is the current release-ready private evaluation surface.
- The bounded stack is intentionally scoped: localhost/private only, bounded domain only, no public API, no production chat, no GPT-like/open-domain assistant claim.
- The live Rust architecture line remains [`instnct-core/examples/neuron_grower.rs`](instnct-core/examples/neuron_grower.rs).
- The 091-100 open-vocab assistant line is a separate runner-local research track. It may inform future training, but it does not mutate or replace the 099 bounded release baseline.
- The Python `graph.py` lane remains historical reference/support, not the public mainline.

### Evidence snapshot

- **099 bounded release-ready gate** passed: fresh local/private deployment harness smoke, SDK smoke, bounded chat service smoke, artifact hash verification, checkpoint unchanged, rollback pointer, no training, and upstream proof lineage all positive.
- **100 assistant capability scale probe** passed: a new target-only runner-local research checkpoint improved over the 094 generation baseline while keeping bounded retention and release artifacts unchanged.
- **089B packaged winner proof** passed: the packaged bounded checkpoint, original 080 winner, and deterministic reproduction child matched byte-level checkpoint hash.
- **088 long-run/concurrency gate** passed: 240/240 service-path requests completed with exact audit coverage and no orphan child jobs.
- **087 OOD/red-team gate** passed through `POST /v1/bounded-chat/infer`, not a direct model runner.
- **Bias-free threshold grower** remains a validated Rust architecture surface on `main`; it is not overwritten by the bounded service release or runner-local open-vocab probes.

Raw run dumps, archived sweeps, older scratch tools, and pre-bounded probe runners are preserved at archive tags or the May 19 consolidation archive branch listed in [`ARCHIVE.md`](ARCHIVE.md), not on active `main`.

The canonical evidence summary lives in [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md).

### Experimental branch

- The current next research decision is whether to continue from `100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE` into a fresh assistant eval/failure map, or pause capability training and polish public docs/wiki.
- This remains research-only until fresh open-domain, multi-turn, Hungarian/English, OOD/refusal, collapse, and retention gates are independently positive.

## 5-Minute Proof

### Bounded release-ready checks

```bash
python -m py_compile scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate.py
python -m py_compile scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale.py
python scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale_check.py --check-only
```

These checks validate the generated local `target/` evidence when it is present. Generated private evaluation artifacts and checkpoints are intentionally not committed to Git.

### Rust and Python surfaces

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cargo test -p instnct-core
python -m compileall Python tools
python -m pytest Python/ -q
```

These commands verify:

- the Rust library compiles and tests pass,
- the Python deploy SDK (`Python/`) remains importable/testable,
- current release-readiness checkers still parse their target-only evidence.

## Read Next

- [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — current bounded release baseline and open-vocab capability-track boundary
- [`docs/REPO_CONSOLIDATION_2026_05_19.md`](docs/REPO_CONSOLIDATION_2026_05_19.md) — what was archived/removed during the cleanup pass
- [`VALIDATED_FINDINGS.md`](VALIDATED_FINDINGS.md) — canonical evidence summary
- [`docs/research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RESULT.md`](docs/research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RESULT.md) — bounded local/private release-ready result
- [`docs/research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_RESULT.md`](docs/research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_RESULT.md) — current capability-scale research result
- [`Python/block_a_byte_unit/README.md`](Python/block_a_byte_unit/README.md) + [`Python/block_b_merger/README.md`](Python/block_b_merger/README.md) — deploy SDK per-block entry
- [VRAXION architecture page (INSTNCT)](https://github.com/VRAXION/VRAXION/wiki/INSTNCT-Architecture)

## License

- Community source license: [LICENSE](LICENSE)
- Commercial royalty terms: [legal/LEGAL.md](legal/LEGAL.md)
- VRAXION Forever Prize charter: [legal/VRAXION_FOREVER_PRIZE_CHARTER.md](legal/VRAXION_FOREVER_PRIZE_CHARTER.md)
- Contributor terms: [CONTRIBUTING.md](CONTRIBUTING.md)
- Citation: [CITATION.cff](CITATION.cff)

VRAXION is **source-available**, not OSI-approved open source. Free community,
research, nonprofit, self-hosted, and internal use are allowed under the
license. Monetized third-party access to VRAXION-powered functionality requires
**19% of Attributable Net Revenue**, allocated as **1% Founder Allocation + 18%
VRAXION Forever Prize Allocation** until a Founder Redirect Event. After that
event, the full **19%** goes to the Prize.

The software license does not grant rights to use the **VRAXION** or **INSTNCT** names, logos, or brand assets except as described in [legal/LEGAL.md](legal/LEGAL.md).
