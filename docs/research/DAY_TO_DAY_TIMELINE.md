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

### 2026-05-02 - D20 output-anchor synthesis and D21A A-block

- D20 deep research intake concluded that the D19 blocker is not raw context
  capacity. D18/D19 already showed context is reachable. The blocker is adding
  context while preserving `top_01` behavior.

```text
recommended D20 objective:
real_context_margin
- max(fake_context_gains)
- output_anchor_divergence_vs_top01
- smooth/accuracy/unigram/echo safety penalties
```

- D21A implemented a scratch reciprocal byte A-block:

```text
byte -> 8 visible bits -> 16D abstract code -> 8 bit logits -> byte logits
decoder = encoder.T
```

- D21A result:

```text
D21A_RECIPROCAL_ABLOCK_PASS
main candidates scanned: 29,718
gate-pass candidates: 1,618
best: 16D redundant_copy_2x, 16 reciprocal edges
exact_byte_acc: 100%
bit_acc: 100%
byte_margin_min: +4.0
single_edge_drop_mean_bit: 100%
```

- D21A crystallize:

```text
compact gate-pass candidate: 14 reciprocal edges
exact_byte_acc: 100%
bit_acc: 100%
byte_margin_min: +2.0
single_edge_drop_mean_bit: 0.991071
```

Interpretation: D21A solves the base byte round-trip gate cleanly. The next
quality step is not more reconstruction accuracy, but error-correcting byte-code
geometry and then a context lane that does not break reconstruction.

Updated source docs:

- `docs/research/PHASE_D20_OUTPUT_ANCHOR_RESEARCH_SYNTHESIS.md`
- `docs/research/PHASE_D21A_RECIPROCAL_ABLOCK_BYTE_ENCODER.md`

### 2026-05-02 - D21B context-extended A-block

- D21B added a sparse context lane on top of the fixed D21A reciprocal byte
  lane. The invariant was:

```text
zero context => D21A behavior remains exact
real context => can steer output byte
fake/shuffled/random context => should not match real context
```

- D21B result:

```text
D21B_CONTEXT_PASS
best confirmed context_dim: 16
context_edges: 16
context_target_count: 256
context_capacity_bits: 8.0
zero_exact_byte_acc: 100%
zero_bit_acc: 100%
real_context_success: 100%
fake_context_success: 0.4718%
context_selectivity: 0.995282
real_context_margin_min: +12.0
```

- Atlas result:

```text
4D context: insufficient for full byte steering
8D context: full pass
16D context: full pass, best high-margin confirmed lane
```

Interpretation: D21A solved the byte adapter; D21B shows a clean context write
channel can control the byte output without breaking zero-context
reconstruction. This does not make a release model yet, but it gives D21C a
clean shell:

```text
A-block byte IO + context lane + tiny recurrent/core block
```

Updated source doc:

- `docs/research/PHASE_D21B_CONTEXT_ABLOCK.md`

### 2026-05-02 - D21C tiny recurrent A-block core

- D21C attached a tiny recurrent/state core behind the D21A/D21B A-block shell.
  The first task was previous-byte recall:

```text
input sequence: [A, B]
target output:  A when B arrives
```

- D21C result:

```text
D21C_PREV_BYTE_CORE_PASS
best high-margin core: state_dim=16, core_edges=16
prev_byte_exact_acc: 100%
long_sequence_exact_acc: 100%
prev_byte_margin_min: +12.0
reset_each_token_acc: 0.3906%
time_shuffle_state_acc: 0.4105%
random_state_acc: 0.3967%
current_byte_cheat_rate: 0%
zero_context_byte_reconstruction_acc: 100%
```

- D21C crystallize also found a compact full-pass core:

```text
state_dim=16
core_edges=8
prev_byte_exact_acc: 100%
long_sequence_exact_acc: 100%
prev_byte_margin_min: +4.0
```

Interpretation: the A-block path now has a complete validated micro-loop:

```text
byte input -> fixed byte lane -> tiny recurrent state -> context lane -> output byte
```

This is not a release model. It is the first clean proof that the A-block shell
can be driven by a state-carrying core rather than only by externally supplied
context.

Updated source doc:

- `docs/research/PHASE_D21C_TINY_RECURRENT_ABLOCK_CORE.md`

### 2026-05-02 - D21D marker-memory A-block core

- D21D moved the A-block core from one-step previous-byte memory to delayed
  marker memory:

```text
input:  MARKER, PAYLOAD, distractor..., QUERY
target: PAYLOAD at QUERY
```

- D21D result:

```text
D21D_MARKER_MEMORY_PASS
fresh oracle:
  eval_sequences: 65536
  distractor_lengths: 1,2,4,8,16
  query_payload_exact_acc: 100%
  query_payload_margin_min: +12.0
  payload_state_collision_count: 0

bounded confirm:
  eval_sequences: 32768
  distractor_lengths: 1,2,4,8,16,32
  pass_count: 12 / 16
  best high-margin core: state_dim=32, memory_edges=16
  query_payload_exact_acc: 100%
  long_sequence_payload_acc: 100%
  non_query_byte_reconstruction_acc: 100%
  reset/time-shuffle/marker-shuffle controls: 0%
  random_state_acc: ~0.31%
  current_byte_cheat_rate: 0%
```

- Compact confirmed family:

```text
identity_memory, 8 edges
query_payload_exact_acc: 100%
query_payload_margin_min: +4.0
random-state control: ~0.34-0.41%
```

- Rescope note: the original full atlas/crystallize shape was too slow, so the
  evidence run was bounded. This is component-level proof, not a release model
  or H512/H8192 unlock.

Interpretation:

```text
The A-block core can store a marked payload,
carry it through distractors,
and recall it only on query.
```

Updated source doc:

- `docs/research/PHASE_D21D_MARKER_MEMORY_ABLOCK_CORE.md`

### 2026-05-02 - D21E multi-slot memory A-block core

- D21E moved D21D from one delayed memory slot to slot-addressed key-value
  memory:

```text
MARKER_A -> PAYLOAD_A
MARKER_B -> PAYLOAD_B
MARKER_C -> PAYLOAD_C
MARKER_D -> PAYLOAD_D
distractor...
QUERY_B -> PAYLOAD_B
```

- D21E result:

```text
D21E_MULTISLOT_MEMORY_PASS
oracle:
  slot_counts: 2,4
  distractor_lengths: 1,2,4,8,16
  eval_sequences: 65536
  query_payload_exact_acc: 100%
  query_shuffle_acc: 0%
  wrong_slot_recall_rate: 0%

bounded confirm:
  eval_sequences: 32768
  distractor_lengths: 1,2,4,8,16,32
  pass_count: 10 / 16
  best high-margin core: state_dim=64, slot_count=4, memory_edges=64
  query_payload_exact_acc: 100%
  query_payload_margin_min: +12.0
  reset/time/marker/query shuffle controls: 0%
  random_state_acc: ~0.19%
  wrong_slot_recall_rate: 0%
```

- Crystallize result:

```text
64-edge high-margin reference -> 32-edge compact confirmed core
compact query_payload_exact_acc: 100%
compact query_payload_margin_min: +4.0
verdict: D21E_MULTISLOT_MEMORY_PASS
```

- Rescope note: full `samples=256` atlas and `131072` confirm were too slow, so
  the run was intentionally bounded to `samples=8` and `32768` confirm. This is
  component-level evidence, not a release model or high-H unlock.

Interpretation:

```text
The A-block core can remember several named byte values,
ignore distractors,
and recall the value belonging to the queried name.
```

Updated source doc:

- `docs/research/PHASE_D21E_MULTISLOT_MEMORY_ABLOCK_CORE.md`

### 2026-05-02 - D22 byte-to-word embedder

- D22 composed 8 parallel D21A A-blocks into a fixed 8-byte window surface:

```text
8 bytes -> 8 parallel A-block codes -> 128D word-ish code -> 8 bytes
```

- D22 result:

```text
D22_COMPACT_WORD_EMBEDDER_PASS
eval_windows: 65536

128D robust reference:
  window_exact_acc: 100%
  byte_margin_min: +4.0
  single_dim_drop_mean_window_exact: 100%
  int8 window vector: 128 bytes
  byte LUT: 4096 bytes

64D compact candidate:
  window_exact_acc: 100%
  byte_margin_min: +2.0
  single_dim_drop_mean_bit: 99.1972%
  int8 window vector: 64 bytes
  byte LUT: 2048 bytes

32D control:
  window_exact_acc: 0.2%
  verdict: fail
```

Interpretation:

```text
A-block = one byte IO/memory cell
D22     = 8-byte window embedder
```

Recommendation: use `128D` as the safer research width and `64D` as the compact
exact deployment candidate.

Updated source doc:

- `docs/research/PHASE_D22_BYTE_WORD_EMBEDDER.md`
