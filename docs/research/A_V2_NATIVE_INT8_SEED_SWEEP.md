# A-v2 Native Int8 Seed Sweep

Date: 2026-05-03

## Verdict

```text
A_V2_INT8_MARGIN_STRONG_GEOMETRY_NEAR_MISS
margin reached 4.000, but geometry 0.758 is below the 0.760 strong gate
```

## What Ran

Implemented `tools/_scratch/a_v2_native_int8_seed_sweep.py`, a hidden-only
A-block search that mutates native `int8_q6` q-values:

```text
storage: int8_q6
scale: 64
weight = q / 64
direct visible->A highway: forbidden
decoder: transpose chain
```

The first full 32-seed plan was too slow with the full geometry evaluator, so
the executed run was rescaled after smoke timing:

```text
hidden_dim: 8, 12, 16
seeds: 20260503..20260505
arms: current_int8_polish, hidden_bridge_restart, random_hidden_restart
steps: 160
workers: 4
```

## Result

Best confirmed native int8 candidate:

```text
artifact: tools/a_v2_hidden_natural_int8_candidate.json
hidden_dim: 12
hidden_in_edges: 13
hidden_out_edges: 34
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: +4.000
ascii_class_geometry: 0.758
effective_copy_penalty: 0.0
single_edge_drop_mean_bit: 0.999
```

Verifier:

```powershell
python tools\a_hidden_natural_int8_artifact.py --verify-artifact tools\a_v2_hidden_natural_int8_candidate.json
```

returns:

```text
A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS
```

## Interpretation

Compared to the previous hidden-natural candidate:

```text
previous A-HiddenNaturalMarginPolish-int8:
  margin:   +3.516
  geometry:  0.770

new A-v2 H12 int8 candidate:
  margin:   +4.000
  geometry:  0.758
```

So this run found the missing margin, but paid a small geometry cost. It is not a
full replacement candidate because the strong geometry gate is `>= 0.760`.

## Decision

```text
A-StableCopy16 remains shipped/default.
A-HiddenNaturalMarginPolish-int8 remains the best geometry A_v2 candidate.
A-v2 H12 int8 candidate is the best margin-safe non-copy candidate.
```

Next recommended work is not more blind seeds. The useful next step is a narrow
polish around the H12 near-miss, explicitly optimizing:

```text
geometry +0.002
while keeping margin >= +4.0
and copy penalty <= 0.10
```
