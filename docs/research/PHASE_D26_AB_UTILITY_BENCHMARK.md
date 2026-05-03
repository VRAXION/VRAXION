# Phase D26 - AB Utility Benchmark

## Verdict

```text
D26_AB_HAS_COMPONENT_UTILITY
```

D26 tested whether the A/B stack is useful to a core, not just able to
roundtrip bytes.

The benchmark compared the same tasks on four surfaces:

```text
raw64        direct 8-byte signed bit lanes
a128         D21/D22 A-window lanes
b64          AB codec latent
b64_composed D24 transform + D25 memory composed over B64
```

## Main Result

Main run:

```text
eval_windows: 65,536
slot_counts: 2,4
distractor_lengths: 1,2,4,8,16,32
control_repeats: 2
artifact: tools/ab_window_codec_v1.json
```

All surfaces solved the exact tasks, and all controls stayed clean:

```text
surface       task                 exact  max_control  edges       state
a128          memory_transform     1.000  0.000        384-640     512
b64           memory_transform     1.000  0.000        192-320     256
b64_composed  memory_transform     1.000  0.000        192-320     256
raw64         memory_transform     1.000  0.000        192-320     256
```

Interpretation:

```text
B64 does not beat RAW64 on pure bit-level tasks.
B64 does beat A128 on width/state/edge count.
B64_composed cleanly connects D24 transform + D25 memory.
```

That is component utility, not semantic-compression proof.

## What This Means

D26 answers the "are we doing anything useful?" question as follows:

```text
No:
  AB/B64 is not yet a better-than-raw semantic abstraction.

Yes:
  AB/B64 is a cleaner internal interface than A128.
  It supports exact composition of memory + transform modules.
```

So the right next step is not "make C deeper because deeper is better". The
right next step is to use B64 as a stable working interface and test stronger
composition:

```text
store X
query reverse(X)
query rotate(X)
query rule-select(copy/reverse/rotate)
```

## Caveat

The originally requested full main shape included `control_repeats=8`. That
variant was too slow for the interactive budget and was rescaled to
`control_repeats=2` after smoke showed deterministic clean controls. The main
still used 65,536 eval windows and the full surface/task matrix.

## Artifacts

```text
output/phase_d26_ab_utility_benchmark_20260503/smoke/
output/phase_d26_ab_utility_benchmark_20260503/main/
output/phase_d26_ab_utility_benchmark_20260503/robustness/
```

Tracked implementation:

```text
tools/_scratch/d26_ab_utility_benchmark.py
```
