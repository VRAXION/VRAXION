# Phase D21B Context-Extended A-Block

Date: 2026-05-02

## Summary

D21B extends the D21A reciprocal byte A-block with a separate sparse context
lane. The byte lane remains fixed and reciprocal:

```text
byte -> 8 visible bits -> 16D byte code -> 8 bit logits -> byte logits
decoder = encoder.T
```

The context lane is tested as an output steering/write channel:

```text
target byte context -> context lane -> output byte bias
```

The important invariant is that zero-context behavior must remain exactly D21A:

```text
zero_context exact_byte_acc = 1.0
zero_context bit_acc = 1.0
zero_context reciprocity_error = 0
```

Implementation:

```text
tools/_scratch/d21b_context_ablock.py
```

## Result

Verdict:

```text
D21B_CONTEXT_PASS
```

All-pair confirm of the top mutated context lane:

```text
context_dim: 16
context_edges: 16
context_target_count: 256
context_capacity_bits: 8.0

zero_exact_byte_acc: 1.0
zero_bit_acc: 1.0
zero_byte_margin_min: +4.0
zero_hidden_collisions: 0
zero_reciprocity_error: 0.0
single_edge_drop_mean_bit: 1.0

real_context_success: 1.0
real_context_margin_mean: +12.0
real_context_margin_min: +12.0

shuffle_context_success: 0.004718
random_context_success: 0.003202
no_context_target_success: 0.0
fake_context_success: 0.004718
context_selectivity: 0.995282
context_margin_selectivity: +28.062746

small_noise_context_bit_acc: 0.990638
```

Best confirmed context entries:

```text
0:0:4 1:1:4 2:2:4 3:3:4 4:4:4 5:5:4 6:6:4 7:7:4
8:0:-4 9:1:-4 10:2:-4 11:3:-4 12:4:-4 13:5:-4 14:6:-4 15:7:-4
```

Interpretation: a 16D context lane can address all 256 byte targets and steer
the A-block output to the requested byte, while fake/shuffled/random context
controls stay near chance. Zero-context byte reconstruction remains perfect.

## Atlas

Atlas result:

```text
D21B context heatmap: brighter = real_context_success * selectivity
legend: PASS=P WEAK=W FAKE=F NONE=.
ctx\edge     4     8    16    32
       4     .     .     .     .
       8 @   P @   P @   P @   P
      16 @   P @   P @   P @   P
```

Meaning:

- `4D` context is insufficient for full byte-level steering.
- `8D` context is already enough to steer all 256 byte targets with the
  identity context map.
- `16D` context provides a redundant high-margin lane and was the best confirmed
  version after mutate/polish.

## Why This Matters

D21A solved byte round-trip IO. D21B shows that a separate context lane can
control the byte answer without breaking the IO adapter. This is component-level
progress toward the missing context-carry behavior:

```text
stable byte adapter
+
explicit context write channel
=
usable shell for a tiny recurrent/core block
```

D21B is not a language model or release checkpoint. It is a clean microcomponent
that can be used in D21C:

```text
A-block byte IO + context lane + tiny recurrent core
```

## Commands

Baseline:

```powershell
python tools\_scratch\d21b_context_ablock.py --mode baseline-check --d21a-json output\phase_d21a_reciprocal_ablock\repo_smoke\ablock_top.json --out output\phase_d21b_context_ablock\baseline
```

Atlas:

```powershell
python tools\_scratch\d21b_context_ablock.py --mode context-atlas --visible 8 --code-dim 16 --context-dims 4,8,16 --context-edge-budgets 4,8,16,32 --samples 512 --workers 10 --eval-pairs 8192 --out output\phase_d21b_context_ablock\atlas
```

Mutate/polish:

```powershell
python tools\_scratch\d21b_context_ablock.py --mode mutate-context --visible 8 --code-dim 16 --context-dim 16 --context-edge-budget 16 --start-family redundant_context --max-steps 500 --workers 10 --stop-after-stale 512 --eval-pairs 8192 --out output\phase_d21b_context_ablock\mutate
```

All-pair confirm:

```powershell
python tools\_scratch\d21b_context_ablock.py --mode confirm --candidates output\phase_d21b_context_ablock\mutate\context_ablock_candidates.csv --top-k 16 --eval-pairs 65536 --out output\phase_d21b_context_ablock\confirm
```

## Tests

Passed:

```text
python -m py_compile tools\_scratch\d21b_context_ablock.py
python tools\check_public_surface.py
D21B baseline-check
D21B context-atlas
D21B mutate-context
D21B all-pair confirm
```

Generated `output/` remains uncommitted.

## Next Step

D21C should attach a tiny recurrent/core block behind the A-block:

```text
input byte -> A-block byte lane -> tiny state/core -> context lane -> output byte
```

D21C acceptance should require:

```text
byte reconstruction remains exact when no context write is requested
state/core can emit context vectors that solve a simple byte-sequence task
fake/shuffled context controls fail
```
