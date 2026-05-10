# ANCHOR-MINI-004 VRAXION Sparse Carrier Contract

## Summary

ANCHOR-MINI-004 tests whether the ANCHOR-MINI-003 shortcut-flip signal survives
when the carrier is a deterministic sparse mutation-selection system instead of
a tiny backprop MLP.

This is not a natural-language AnchorCell test and not a grounding claim. It is
a carrier-boundary probe:

```text
Does decomposed AnchorCell-style process supervision help OOD shortcut
resistance when the final decision is routed through a sparse process carrier?
```

## Dataset

Use the same MINI-003 logical data contract:

- 4 candidates.
- 4 goal/effect categories.
- Symbolic goal/effect IDs.
- Train shortcut: highest `surface_prior` points to gold with high probability.
- Eval shortcut flip: highest `surface_prior` points to a wrong candidate with
  high probability.

The Python orchestrator writes per-seed datasets using the existing
`run_anchor_mini003.py` generator. The Rust carrier reads those frozen datasets.

## Carrier Arms

```text
SPARSE_DIRECT
  Final answer is produced by a sparse direct surface path.

SPARSE_AUX_DIRECT
  A process branch is trained, but final answer remains direct.

SPARSE_ROUTED
  Final answer is produced by the process/match branch.

SPARSE_HYBRID
  Final answer combines direct and process branches.

SPARSE_SHUFFLED_ROUTED
  Routed branch uses shuffled process labels as a negative control.
```

## Sparse Mutation Carrier

The carrier is an audit-sized sparse signed feature model evolved by
mutation-selection. It is intentionally small and deterministic. It tests
routing, not the full production recurrent grower.

Allowed feature families:

```text
surface feature: candidate surface prior
match feature: candidate matches goal/effect category
shuffled match feature: shuffled process-control target
bias feature: shared bias
```

Decision routing:

```text
direct carriers: final answer from direct sparse branch
routed carriers: final answer from process sparse branch
hybrid carriers: final answer from direct + process sparse branches
```

Fitness is smooth:

```text
answer_score = mean softmax probability assigned to gold answer
process_score = mean sigmoid process-bit likelihood
total_score = answer_score + aux_weight * process_score where applicable
```

Eval-only reporting includes shortcut trap rate. Trap penalty is not used as a
train signal.

## Pass / Fail

The stress is invalid if:

```text
train surface shortcut alignment < 0.85
eval surface shortcut flip rate < 0.85
```

Primary positive requires valid stress and:

```text
SPARSE_ROUTED OOD accuracy >= SPARSE_DIRECT + 0.25
SPARSE_ROUTED OOD accuracy >= SPARSE_SHUFFLED_ROUTED + 0.25
SPARSE_ROUTED shortcut trap rate <= 0.25
SPARSE_AUX_DIRECT does not match SPARSE_ROUTED unless final decision uses process state
SPARSE_HYBRID remains directionally positive
valid seeds >= 50
```

Strong status:

```text
ANCHOR_MINI_004_VRAXION_SPARSE_STRONG_POSITIVE
```

Other statuses:

```text
ANCHOR_MINI_004_VRAXION_SPARSE_WEAK_POSITIVE
ANCHOR_MINI_004_VRAXION_SPARSE_NEGATIVE
ANCHOR_MINI_004_VRAXION_SPARSE_INVALID_STRESS
ANCHOR_MINI_004_VRAXION_SPARSE_RESOURCE_BLOCKED
```

## Parallel Sweep

Default full sweep:

```text
seeds: 2026-2125
jobs: 16
RAYON_NUM_THREADS: 1 per job
outputs: target/anchorweave/anchor_mini004_vraxion_sparse/
```

The orchestrator is append-only and resumable:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
jobs/<carrier_seed_config>/
```

## Claim Boundary

A positive result means the AnchorCell process signal transfers to this
audit-sized sparse mutation-selection carrier when the decision is routed
through process state. It does not prove Qwen behavior, natural-language
AnchorCells, full INSTNCT recurrent behavior, or symbol grounding at scale.
