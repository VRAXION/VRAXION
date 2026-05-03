# VRAXION Research Scoreboard

Last updated: 2026-05-03

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
A-HiddenNaturalMarginPolish is the A_v2 strong candidate, not yet the shipped A-block.
```

## A-Block Candidates

| Name | Old Alias | What It Is | Exact | Margin | Geometry | Size / Edges | Status |
|---|---:|---|---:|---:|---:|---:|---|
| `A-StableCopy16` | D21A | Direct redundant byte-to-A16 codec | 100% | +4.0 | 0.669 | 16 direct | **Current shipped/default** |
| `A-HiddenBitGain16` | D21I | Hidden-only A, mostly one-bit gain | 100% | +4.0 | 0.731 | 26 hidden | Useful hidden lead, still copy-like |
| `A-HiddenNatural16` | D21J | Hidden-only non-copy natural A | 100% | +2.5 | 0.777 | 37 hidden / 29 effective | superseded by polished candidate |
| `A-HiddenNaturalMarginPolish` | A_v2 polish | Hidden-only non-copy A with stronger margin | 100% | +3.516 | 0.770 | 38 hidden / 30 effective | **A_v2 strong candidate** |
| `A-NaturalSparse16` | D21G | Direct natural sparse A | 100% | +2.5 | 0.764 | 28 effective | Research-only baseline for A_v2 |
| `Legacy-C19-H24` | old L0 byte unit | Extra-hidden C19 byte codec | 100% | +10.537 | 0.905 | 416 | Strong reference, not current reciprocal A |
| `Binary-C19-H16` | old binary C19 | Smaller C19 byte codec | 100% | +0.002 | 0.905 | 384 | Reference only |

Decision:

```text
Use now:
  A-StableCopy16

Improve next:
  A-v2 AB compatibility check using A-HiddenNaturalMarginPolish
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
| `A-HiddenNaturalMarginPolish` | 100% | +3.516 | 0.770 | 7.8 | 0.00 | 29.15 | best natural byte geometry after polish |
| `A-NaturalSparse16` | 100% | +2.5 | 0.764 | 7.8 | 0.00 | 28.85 | strong direct natural reference |
| `A-HiddenBitGain16` | 100% | +4.0 | 0.731 | 7.9 | 0.94 | 27.17 | robust but still copy-like |
| `A-StableCopy16` | 100% | +4.0 | 0.669 | 8.0 | 1.00 | 26.49 | shipped/default, Hamming-like |

Decision:

```text
A-HiddenNaturalMarginPolish wins the revived near/far geometry audit.
A-StableCopy16 remains shipped/default because its decode margin is safer.
Next A work is AB compatibility testing before replacement.
```

## AB / B Surface

| Name | Old Alias | What It Proves | Main Result | Status |
|---|---:|---|---|---|
| `AB-WindowCodec64` | AB V1 / D23 | 8-byte window <-> B64 | 65,536 windows exact, margin +2.0, B collisions 0 | **Stable interface** |
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

## Who Won?

```text
Best current A:
  A-StableCopy16

Best A_v2 research lead:
  A-HiddenNaturalMarginPolish

Best B bus:
  AB-WindowCodec64 / B64

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
1. A-v2 AB compatibility check
   Goal: rebuild/test AB-WindowCodec64 using A-HiddenNaturalMarginPolish.

2. C-StreamTokenizer -> Switchboard integration
   Goal: text stream creates worker calls, not only single 8-byte examples.

3. Decimal ALU formatting
   Goal: 27*852 -> 23004, not only mod256 -> 220.

4. LANG worker
   Goal: replace NO_LANG_WORKER with a real language-processing lane.
```
