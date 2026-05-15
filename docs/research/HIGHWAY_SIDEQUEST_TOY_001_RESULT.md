# HIGHWAY_SIDEQUEST_TOY_001 Result

## Run

```text
seeds=2026,2027
arms=MLP_STATIC,GRU,LSTM,HIGHWAY_ONLY_SIDEPROCESSORS,HIGHWAY_SPARSE_SIDE_LINKS,HIGHWAY_DENSE_SIDE_LINKS,HIGHWAY_PRISMION_SIDEPROCESSORS
widths=16,32
side_counts=4
depths=2
train_examples=2048
eval_examples=2048
epochs=80
jobs=20
completed_jobs=28
```

## Verdict

```json
[
  "STANDARD_RNN_SUFFICIENT"
]
```

## Arm Summary

| Arm | W | K | D | Final | Heldout | Length | Probe | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GRU` | `16` | `0` | `0` | `0.665` | `0.602` | `0.570` | `0.799` | `2185` |
| `GRU` | `32` | `0` | `0` | `0.698` | `0.624` | `0.635` | `0.820` | `7433` |
| `HIGHWAY_DENSE_SIDE_LINKS` | `16` | `4` | `2` | `0.648` | `0.589` | `0.580` | `0.817` | `4909` |
| `HIGHWAY_DENSE_SIDE_LINKS` | `32` | `4` | `2` | `0.679` | `0.591` | `0.635` | `0.829` | `17997` |
| `HIGHWAY_ONLY_SIDEPROCESSORS` | `16` | `4` | `2` | `0.636` | `0.583` | `0.575` | `0.814` | `4909` |
| `HIGHWAY_ONLY_SIDEPROCESSORS` | `32` | `4` | `2` | `0.679` | `0.596` | `0.634` | `0.830` | `17997` |
| `HIGHWAY_PRISMION_SIDEPROCESSORS` | `16` | `4` | `2` | `0.653` | `0.574` | `0.594` | `0.804` | `5037` |
| `HIGHWAY_PRISMION_SIDEPROCESSORS` | `32` | `4` | `2` | `0.688` | `0.609` | `0.656` | `0.837` | `18253` |
| `HIGHWAY_SPARSE_SIDE_LINKS` | `16` | `4` | `2` | `0.648` | `0.569` | `0.592` | `0.810` | `4909` |
| `HIGHWAY_SPARSE_SIDE_LINKS` | `32` | `4` | `2` | `0.696` | `0.613` | `0.657` | `0.834` | `17997` |
| `LSTM` | `16` | `0` | `0` | `0.600` | `0.531` | `0.493` | `0.771` | `2729` |
| `LSTM` | `32` | `0` | `0` | `0.624` | `0.549` | `0.536` | `0.786` | `9545` |
| `MLP_STATIC` | `16` | `0` | `0` | `0.548` | `0.487` | `0.481` | `0.787` | `1513` |
| `MLP_STATIC` | `32` | `0` | `0` | `0.544` | `0.480` | `0.486` | `0.783` | `4041` |

## Interpretation

This is a useful negative/partial result.

The task is not too easy: `MLP_STATIC` stays around `0.48` on heldout/length. The recurrent path matters, because GRU and the highway variants beat static bag-style processing.

But the highway topology is not proven. It does **not** beat the best GRU by the required `+0.10` margin on heldout composition and length generalization.

Best comparison:

```text
GRU w32:
  heldout=0.624
  length=0.635

HIGHWAY_SPARSE w32/k4/d2:
  heldout=0.613
  length=0.657

HIGHWAY_PRISMION w32/k4/d2:
  heldout=0.609
  length=0.656
```

Sparse highway slightly improves length generalization over GRU, but loses on heldout composition. Prismion sideprocessors do not beat matched sparse MLP sideprocessors on the main topology gate.

## Next Implication

Do not treat the highway/sidequest topology as proven yet.

Recommended next options:

1. Run a broader topology grid only if the specific question is whether `K/depth/width` rescues the topology.
2. Otherwise return to the stronger current blocker: `EventOccurredGate` / no-op factuality adapter from public data.
3. Do not build a large Prismion highway cell from this result alone.

## Claim Boundary

This is an abstract symbolic topology probe. It is not an English parser, symbol grounding proof, consciousness claim, or full VRAXION architecture test.
