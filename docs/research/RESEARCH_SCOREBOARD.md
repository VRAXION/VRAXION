# VRAXION Research Scoreboard

Last updated: 2026-05-04

This page is the human-readable scoreboard. Descriptive names are primary; old
`Dxx` phase IDs appear only as aliases.

## Current Winning Chain

```text
byte
  -> A-StableCopy16
  -> AB-WindowCodec64
  -> C-ContentRouter
  -> SwitchboardExecution
      |-> B-LatentTransform
      |-> B-SlotMemory
      '-- ALU-CompactMul / ALU-OpLaneSandwich
```

Current default:

```text
A-StableCopy16 is still the shipped A-block.
A-v2-H12-GeometryPolish is the A_v2 strong candidate.
AB-v2-H12-BitBridge is the compatible AB candidate, not yet the shipped AB codec.
```

## A-Block Candidates

| Name | Old Alias | What It Is | Exact | Margin | Geometry | Size / Edges | Status |
|---|---:|---|---:|---:|---:|---:|---|
| `A-StableCopy16` | D21A | Direct redundant byte-to-A16 codec | 100% | +4.0 | 0.669 | 16 direct | **Current shipped/default** |
| `A-HiddenBitGain16` | D21I | Hidden-only A, mostly one-bit gain | 100% | +4.0 | 0.731 | 26 hidden | Useful hidden lead, still copy-like |
| `A-HiddenNatural16` | D21J | Hidden-only non-copy natural A | 100% | +2.5 | 0.777 | 37 hidden / 29 effective | superseded by polished candidate |
| `A-HiddenNaturalMarginPolish` | A_v2 polish | H8 hidden-only non-copy A with stronger margin | 100% | +3.516 | 0.770 | 38 hidden / 30 effective | superseded by H12 polish |
| `A-v2-H12-GeometryPolish` | A_v2 H12 polish | H12 native int8 non-copy A with stable margin and strong geometry | 100% | +4.0 | 0.829 | 67 structural / 46 effective | **A_v2 strong candidate; AB compatibility pass** |
| `A-NaturalSparse16` | D21G | Direct natural sparse A | 100% | +2.5 | 0.764 | 28 effective | Research-only baseline for A_v2 |
| `Legacy-C19-H24` | old L0 byte unit | Extra-hidden C19 byte codec | 100% | +10.537 | 0.905 | 416 | Strong reference, not current reciprocal A |
| `Binary-C19-H16` | old binary C19 | Smaller C19 byte codec | 100% | +0.002 | 0.905 | 384 | Reference only |

Decision:

```text
Use now:
  A-StableCopy16

Improve next:
  AB-v2 worker regression using AB-v2-H12-BitBridge
```

## A-Space Utilization

| Name | Active A16 Lanes | Intrinsic Rank | PCA 95% Dims | Lane Balance | Class Separation | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `A-StableCopy16` | 16 | 8 | 8 | best/even | 1.126 | uses A16 as even redundant copy |
| `A-HiddenBitGain16` | 16 | 8 | 8 | one lane boosted | 1.156 | confirms hidden bit-gain |
| `A-HiddenNatural16` | 16 | 8 | 8 | less even | 1.319 | best A-branch class separation |
| `A-NaturalSparse16` | 15 | 8 | 8 | uneven | 1.249 | natural sparse direct reference |

Key rule:

```text
Input has 8 bits, so reciprocal A candidates cannot create more than 8
independent information dimensions. Extra A lanes are used for redundancy,
weighting, and geometry.
```

## A-GeometryAuditRevival

This revives the old byte-embedding similarity audit:

```text
Does A16 only roundtrip bytes, or do related bytes land close together?
```

| Name | Exact | Margin | Geometry | Effective Rank | Copy Penalty | Audit Score | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `A-v2-H12-GeometryPolish` | 100% | +4.0 | 0.829 | 7.0 | 0.00 | TBD | best current A_v2 candidate; AB compatibility pass |
| `A-HiddenNaturalMarginPolish` | 100% | +3.516 | 0.770 | 7.8 | 0.00 | 29.15 | best natural byte geometry after polish |
| `A-NaturalSparse16` | 100% | +2.5 | 0.764 | 7.8 | 0.00 | 28.85 | strong direct natural reference |
| `A-HiddenBitGain16` | 100% | +4.0 | 0.731 | 7.9 | 0.94 | 27.17 | robust but still copy-like |
| `A-StableCopy16` | 100% | +4.0 | 0.669 | 8.0 | 1.00 | 26.49 | shipped/default, Hamming-like |

Decision:

```text
A-v2-H12-GeometryPolish now wins the practical A_v2 gate: copy penalty 0,
margin +4.0, geometry 0.829.
A-StableCopy16 remains shipped/default until downstream worker regression is
re-run with the H12 A-v2 AB bridge.
```

## AB / B Surface

| Name | Old Alias | What It Proves | Main Result | Status |
|---|---:|---|---|---|
| `AB-WindowCodec64` | AB V1 / D23 | 8-byte window <-> B64 | 65,536 windows exact, margin +2.0, B collisions 0 | **Stable interface** |
| `AB-v2-H12-BitBridge` | A-v2 AB gate | A-v2-H12 A16 <-> canonical B64 byte-bit bus | 65,536 windows exact, B64 semantic 100%, random control fails | **Compatibility pass; not default yet** |
| `B-LatentTransform` | D24 | B64 can do exact transforms | copy/reverse/rotate/bit_not exact, random control 0% | **Worker-ready** |
| `B-SlotMemory` | D25 | B64 addressed memory | 2-slot and 4-slot exact, wrong-slot 0% | **Worker-ready** |
| `AB-UtilityBenchmark` | D26 | Does AB/B help vs RAW/A? | `AB_HAS_COMPONENT_UTILITY` | Useful as composable interface, not magic compression |

Decision:

```text
B64 is a good stable bus/interface.
Its biggest value is composition, not outperforming raw bits on every toy task.
```

## C Router / Switchboard

| Name | Old Alias | What It Does | Main Result | Status |
|---|---:|---|---|---|
| `C-ContentRouter` | D28 | Routes input to LANG/ALU/MEM/TRANSFORM/UNKNOWN | heldout route accuracy 100%; controls about 20% | **Pass** |
| `SwitchboardExecution` | D29 | Runs only selected worker; inactive lanes empty | route/output/lane-empty all 100% | **Pass** |
| `C-StreamTokenizer` | D31A | Tokenizes B64 stream into token events + route hints | token stream exact 100%; ALU calls exact 100% | **Pass** |

Current limitation:

```text
LANG has no real language worker yet.
LANG currently routes to NO_LANG_WORKER.
```

## ALU / Worker Blocks

| Name | Old Alias | What It Does | Main Result | Status |
|---|---:|---|---|---|
| `B-ALU-Router` | D27 | B64 ALU operations | 9 ops exact over 65,536 windows | Pass |
| `ALU-OpLaneSandwich` | D30A | Separate ADD/SUB/MUL/AND/OR/XOR lanes | each op exact; inactive lanes 100% empty | Pass |
| `ALU-CompactMul` | D30B | Compact modulo-256 multiplier | exact 65,536 pairs; 0 table entries; 256x smaller than table | **Best MUL lane** |

Important:

```text
27*852 currently means byte/mod256 output:
  27 * 852 mod 256 = 220

Full decimal output:
  27 * 852 = 23004
is not implemented yet.
```

## Memory Primitives

| Name | Old Alias | Task | Result | Status |
|---|---:|---|---|---|
| `A-PrevByteMemory` | D21C | output previous byte | exact all-pair prev-byte memory | Component proof |
| `A-MarkerMemory` | D21D | marker, payload, distractors, query -> payload | exact marker recall | Component proof |
| `A-MultiSlotMemory` | D21E | multiple addressed slots | multi-slot memory pass | Component proof |
| `B-SlotMemory` | D25 | addressed B64 memory | exact 2/4 slot B64 recall | Current memory worker |

## Candidate Primitives

| Name | Old Alias | What It Is | Verdict | Use |
|---|---:|---|---|---|
| `VoltageKnobNeuron` | D32 | voltage/energy splitter primitive | useful proof surface | candidate primitive |
| `AngleKnobNeuron` | D32B/D32C | angle-based routing activation | routing-only winner, not universal activation | possible router primitive |

## Mechanism Toy Notes

Small refraction probes remain research-only, but the current best read is:

```text
frame pointer + recurrent attractor / authority switching
```

Frequency embedding ablation:

- `fixed_sincos` gives a small consistent improvement over the learned/random-vector baseline on authority-switch and dog-separation metrics.
- More elaborate phase modes are not necessary in the current toy setup.
- The current winner remains frame pointer plus recurrent attractor / authority switching, not explicit frequency embedding geometry.
- Future wave/interference work should focus more on recurrent dynamics and edge/node modulation than on token embedding geometry alone.

Raw wave pointer ablation:

- `token_wave` is a weak positive over `none` on accuracy, refraction, authority-switch, mean actor-switch, and dog-switch metrics.
- `pointer_resonance` and `pointer_resonance_signed` are sensitive to wrong/frozen/shuffled pointer interventions, but they do not beat the simpler `token_wave` mode on authority switching.
- Explicit pointer/neuron resonance is not required by the current toy tasks.

FlyWire / topology-prior ablation:

- A local `/home/deck/work/flywire/mushroom_body.graphml` sample exists and can be used as a small recurrent-mask prior.
- At matched edge budget on `latent_refraction`, `hub_rich` beats `random_sparse` on accuracy, recurrence gain, final refraction index, authority-switch score, and seed variance.
- `flywire_sampled` does not beat `random_sparse` on authority/refraction in the first pass, though it reduces seed variance.
- Current interpretation: hub/heavy-tail structure is the live topology signal; specific mushroom-body wiring is not yet supported as better than random sparse.

Hub-rich validation:

- `hub_rich` validates as task-specific positive on `latent_refraction` at update rates `0.2` and `0.3`.
- `hub_rich` does not beat `random_sparse` on the stricter `multi_aspect_token_refraction` grid.
- Hub nodes are load-bearing inside trained hub-rich models: top-10% hub ablation hurts far more than same-count random node ablation across both tasks.
- Current topology read: hubs can help group-level latent refraction, but they are not a universal replacement for random sparse masks.

Hub degree-preserving control:

- Training degree-preserving shuffled hub masks from scratch recovers much of the hub benefit, and at `latent_refraction/update=0.2` beats both `random_sparse` and the sampled `hub_rich` mask.
- At `latent_refraction/update=0.3`, the original `hub_rich` mask remains strongest on final refraction and authority-switch metrics, so specific hub wiring can still matter.
- On `multi_aspect_token_refraction`, degree-preserving hub masks are better than the sampled `hub_rich` mask and roughly match `random_sparse`, but they do not establish a universal hub prior.
- FlyWire degree-preserving random masks generally beat raw FlyWire-sampled masks, weakening exact-FlyWire-wiring claims in this toy setup.
- Current refined topology read: degree concentration is often the useful part, exact sampled wiring is task/regime dependent, and no biology/FlyWire claim is supported.

Inferred frame pointer:

- `inferred_frame_pointer` tests the next step after explicit frame tokens: predict the frame from the input bundle, then use that predicted frame as the recurrent pointer.
- First run: frame prediction is clean (`1.0`) and predicted-frame task accuracy (`0.881`) is close to oracle-frame accuracy (`0.893`).
- Recurrence remains load-bearing: zero-recurrent accuracy is `0.587`, randomized-recurrent accuracy is `0.525`, and random-label control is `0.503`.
- Pointer-specific necessity is still unclear because frame-head-only (`0.853`) and no-frame (`0.868`) baselines remain fairly strong; wrong-forced-frame drops to `0.826`, a real but not decisive hit.
- Current read: inferred frame selection works in the toy, but a stricter bottleneck/cue test is needed before claiming the predicted pointer is the dominant authority-switch path.

Query-cued frame pointer:

- `query_cued_frame_pointer` reuses the same observation under multiple toy query cues, so the query rather than salience determines the target frame.
- First run: frame prediction is perfect (`1.0`), predicted-pointer accuracy is `0.680`, no-query baseline is lower (`0.603`), query ablation/shuffle hurt (`0.571`/`0.565`), randomized recurrent is near chance (`0.513`), and random-label control is `0.513`.
- The no-pointer query baseline is close on accuracy (`0.669`) and only slightly weaker on authority/refraction (`0.022`/`0.0115`) than the pointer path (`0.050`/`0.0325`).
- Current read: query-cued frame selection works and recurrence/query dependence are real, but pointer-specific necessity is not supported; query conditioning alone is sufficient in this toy.

## Who Won?

```text
Best current A:
  A-StableCopy16

Best A_v2 research lead:
  A-v2-H12-GeometryPolish

Best B bus:
  AB-WindowCodec64 / B64 current default
  AB-v2-H12-BitBridge current A_v2-compatible candidate

Best transform worker:
  B-LatentTransform

Best memory worker:
  B-SlotMemory

Best router:
  C-ContentRouter

Best switchboard:
  SwitchboardExecution

Best ALU MUL:
  ALU-CompactMul
```

## Next Sensible Improvements

```text
1. AB-v2 worker regression
   Goal: prove B-LatentTransform, B-SlotMemory, ALU, C-router, and Switchboard
   still pass when input/output A surface is A-v2-H12.

2. C-StreamTokenizer -> Switchboard integration
   Goal: text stream creates worker calls, not only single 8-byte examples.

3. Decimal ALU formatting
   Goal: 27*852 -> 23004, not only mod256 -> 220.

4. LANG worker
   Goal: replace NO_LANG_WORKER with a real language-processing lane.
```
