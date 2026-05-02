# VRAXION Day-to-Day Research Timeline

Last updated: 2026-05-02

This file is the compact public sync surface for the current D-series research
thread. It is intentionally conservative: finished gates are marked as done,
running experiments are marked as running, and generated `output/` data remains
untracked.

## Current Snapshot

```text
Release-ready AI path:
[========__] ~78%

[1] H384 trusted research checkpoint
    DONE: D10u/D13 top_01 packaged, SHA256 recorded, 16k/30-seed gate passed

[2] High-H brute force
    BLOCKED: D15B projection selectivity controls rejected H16384 raw signals

[3] Context-carrying capability
    CURRENT: D16b main context-climb running
    status: multiple accepted context-signal candidates observed, not yet confirmed

[4] Next release unlock
    D16 context gate + artifact/state gate on top D16b candidates
```

## Day-by-Day Timeline

### 2026-04-29 - D9/D10 basin mapping foundation

- D9/D10 work established that H384 seed2042 contains real local landscape
  signal rather than pure random noise.
- Causal analysis identified edge + threshold co-adaptation as the working
  mechanism behind the best H384 basin candidates.
- The research direction moved from "find any smooth gain" to "prove gains
  survive artifact, state-shuffle, and multi-objective controls."

### 2026-04-30 - D10 release-readiness hardening

- D10r tightened the evaluation stack around artifact/null controls.
- The older beta.8 generalist checkpoint remained a real research finding, but
  failed the stricter state-identity path and was not kept as the release-track
  checkpoint.
- D10u moved D10r-v8 controls into the search loop and produced the H384
  `top_01` state-anchored candidate.

### 2026-05-01 - D13 packaged `top_01`

- D13 copied the D10u top checkpoint to a stable local release artifact path:

```text
output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt
```

- Checksum:

```text
b76789c42f4349ee28c18ce97bc5f0811a89c9b138e6ecdb86fa55626f019ddb
```

- Promotion-grade prior evidence:

```text
D10U_TOP01_16K_SHARDED_PASS
eval_len=16000
fresh seeds=30
pass shards=30/30
min trusted_mo_ci_low=+0.0844932857
blocking controls=none
```

- Limitation recorded by `chain_diagnosis`:

```text
Context-dependent predictions: 0/4
```

Interpretation: `top_01` is the best trusted H384 research checkpoint, but not
yet a production language-capability checkpoint.

### 2026-05-02 - D15B high-H blocked, D16 context path opened

- D15B checked H16384 / 400k-edge high-H scout signals with candidate-level
  projection selectivity.
- Result:

```text
D15B_PROJECTION_SELECTIVITY_BLOCKED
```

- Meaning: high-H GPU exploration is feasible and reactive, but the current
  projection/readout path still produces control-compatible wins. H512/H8192/
  H16384 brute-force remains blocked for release-candidate purposes.

- D16 established the next useful release-track question:

```text
Can H384 top_01 carry real sequential context?
```

- D16 result on the packaged checkpoint:

```text
D16_CONTEXT_BLOCKED
context-dependent predictions=0/4
```

- D16b smoke added `context-climb` and showed context is locally reachable, but
  the first reachable context behavior traded off existing safety metrics:

```text
D16B_CONTEXT_TRADEOFF
accepted=0/20
context signal candidates=1
tradeoff candidates=8
artifact candidates=10
```

- D16b main is currently running with:

```text
mode=context-climb
H=384
start=top_01 packaged checkpoint
mutation scope=edge,threshold
climbers=12
steps=80
eval_len=1000
eval seeds=974001..974008
```

Current live status at this sync:

```text
D16b main: running
progress: about 70%+
observed: multiple accepted context-signal candidates
final verdict: pending reload/context-gate validation
```

## Current Decision Tree

```text
D16b main completes
        |
        v
Run D16 context gate on top candidates
        |
        |-- passes context + safety + artifact gates
        |      v
        |   D16c margin confirm:
        |   real sequential context must beat reset, time-shuffle,
        |   state-shuffle, random/no-network controls
        |
        |-- context exists but safety tradeoff remains
        |      v
        |   threshold-only context polish + motif-biased local search
        |
        '-- no stable context signal
               v
            redesign context objective/readout before any high-H scaling
```

## Known Blockers

- High-H brute force is not release-relevant until projection/readout controls
  stop producing control-compatible wins.
- The current best H384 checkpoint is artifact-safe but still needs stable
  context-carrying behavior.
- D16b main candidates are not promotable until reload validation, context-gate
  validation, D10r-v8 artifact/state validation, and a longer confirm.

## Source Research Docs

- `docs/research/PHASE_D13_H384_TOP01_RESEARCH_RELEASE_PACKAGE.md`
- `docs/research/PHASE_D15B_PROJECTION_SELECTIVITY_GATE.md`
- `docs/research/PHASE_D16_TOP01_CONTEXT_GATE.md`
- `docs/research/PHASE_D16B_CONTEXT_CLIMB_SMOKE.md`
