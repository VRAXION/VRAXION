# A-v2 H12 Native Int8 Geometry Polish

Date: 2026-05-03

## Verdict

```text
A_V2_H12_INT8_GEOMETRY_POLISH_PASS
```

The H12 near-miss was polished into a strong A-v2 candidate:

```text
Before:
  margin:   +4.000
  geometry:  0.758149
  copy:      0.00

After:
  margin:   +4.000
  geometry:  0.828519
  copy:      0.00
```

## What Changed

Added `tools/_scratch/a_v2_h12_int8_geometry_polish.py`, a narrow native
`int8_q6` search around the H12 A-v2 candidate.

Constraints stayed fixed:

```text
hidden_dim: 12
direct visible->A highway: forbidden
decoder: transpose chain
weights: int8 q / 64
exact byte roundtrip: required
margin >= +4.0: required
copy penalty <= 0.10: required
```

## Confirmed Candidate

Artifact:

```text
tools/a_v2_hidden_natural_int8_candidate.json
```

Verifier:

```powershell
python tools\a_hidden_natural_int8_artifact.py --verify-artifact tools\a_v2_hidden_natural_int8_candidate.json
```

returns:

```text
A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +4.0
hidden_collisions: 0
hidden_dim: 12
hidden_in_edges: 16
hidden_out_edges: 51
```

## Interpretation

This resolves the previous tradeoff:

```text
A-StableCopy16:
  safe margin, weak natural geometry

old A-HiddenNaturalMarginPolish:
  good natural geometry, weaker margin

new A-v2-H12-GeometryPolish:
  safe margin + strong natural geometry
```

It is now the best A-v2 candidate.

Follow-up result: `A_V2_AB_COMPATIBILITY_PASS` showed it can sit under an
AB-style reciprocal B64 bridge while keeping canonical byte-bit B64 semantics.

## Next Gate

AB compatibility is now done. Next gate is downstream worker regression:

```text
B-LatentTransform
B-SlotMemory
ALU-CompactMul / ALU-OpLaneSandwich
C-StreamTokenizer / C-ContentRouter
SwitchboardExecution
```
