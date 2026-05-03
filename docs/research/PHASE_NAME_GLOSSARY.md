# VRAXION Research Name Glossary

Use descriptive names first. Keep old `Dxx` phase IDs only as cross-reference
aliases.

## Current A/AB Pipeline Names

| Use This Name | Old Alias | Meaning |
|---|---:|---|
| `A-StableCopy16` | D21A | Current shipped/minimal byte A-block: byte bits map redundantly into A16. |
| `A-NaturalSparse16` | D21G | Natural sparse A candidate: better ASCII geometry, lower margin, research-only. |
| `A-HiddenBitGain16` | D21I | Hidden-only A candidate that mainly amplifies one ASCII-relevant bit. |
| `A-HiddenNatural16` | D21J | Hidden-only non-copy A candidate: best natural A_v2 lead. |
| `A-HiddenNaturalMarginPolish` | A_v2 polish | Polished hidden-natural A candidate: higher margin while preserving non-copy geometry. |
| `A-SpaceUtilization` | D21K | Diagnostic report for lane energy, PCA, rank, and class separation. |
| `A-GeometryAuditRevival` | old byte-embed-dim / geometry_sweep | Revived near/far byte embedding audit: cosine/distance matrix, clusters, nearest neighbors. |
| `AB-WindowCodec64` | D23 / AB V1 | 8 bytes -> A128 -> B64 -> A128 -> 8 bytes. |
| `B-LatentTransform` | D24 | Exact transforms over B64: copy, reverse, rotate, bit_not. |
| `B-SlotMemory` | D25 | Addressed B64 key-value memory. |
| `AB-UtilityBenchmark` | D26 | RAW/A/B surface utility comparison. |

## Current C/D Worker Names

| Use This Name | Old Alias | Meaning |
|---|---:|---|
| `C-ContentRouter` | D28 | Classifies B64 stream into LANG/ALU/MEM/TRANSFORM/UNKNOWN. |
| `SwitchboardExecution` | D29 | Routes selected input into one worker and keeps inactive lanes empty. |
| `ALU-OpLaneSandwich` | D30A | Separate ADD/SUB/MUL/AND/OR/XOR lanes. |
| `ALU-CompactMul` | D30B | Replaces MUL lookup table with exact compact partial-product multiplier. |
| `C-StreamTokenizer` | D31A | B64 stream tokenizer/embedder with token events and route hints. |

## Memory Primitive Names

| Use This Name | Old Alias | Meaning |
|---|---:|---|
| `A-PrevByteMemory` | D21C | Tiny recurrent core recalls previous byte. |
| `A-MarkerMemory` | D21D | Marker, payload, distractors, query -> payload. |
| `A-MultiSlotMemory` | D21E | Multiple marker/query slots with addressed recall. |

## Candidate Primitive Names

| Use This Name | Old Alias | Meaning |
|---|---:|---|
| `VoltageKnobNeuron` | D32 | Voltage/energy splitter primitive proof. |
| `AngleKnobNeuron` | D32B/D32C | Discrete angle-based routing activation. Useful mostly for routing, not universal activation. |

## H384 / Release Evidence Names

| Use This Name | Old Alias | Meaning |
|---|---:|---|
| `H384-Top01Checkpoint` | D13/top_01 | Trusted H384 research checkpoint. |
| `ArtifactNullGate` | D10r-v5/v8 | Evaluator artifact/state-shuffle control gate. |
| `BasinAtlas` | D14 | H384 basin scan / candidate harvest. |
| `ContextCarrySearch` | D16 | Search for real recurrent context usage. |
| `ContextMarginConfirm` | D16C | Real-vs-fake context margin confirmation. |
| `MarginFirstContextSearch` | D18 | Context search objective based on real-vs-fake margin. |

## Speaking Convention

Preferred:

```text
A-HiddenNatural16 is the A_v2 lead.
```

Allowed cross-reference:

```text
A-HiddenNatural16 (old D21J) is the A_v2 lead.
```

Avoid as primary wording:

```text
D21J is the lead.
```
