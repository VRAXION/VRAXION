# HIGHWAY_POCKET_MUTATION_001 Result

## Status

Implemented and ran the bounded sanity + 3-seed smoke.

This is a protected-highway + gated-sidepocket mutation smoke with a small phase-lock bridge. It is runner-local and does not modify `instnct-core` public APIs.

## Runs

Sanity:

```text
out=target/pilot_wave/highway_pocket_mutation_001/sanity
seeds=2026
steps=200
eval_examples=512
H=128
pockets=4
jackpot=6
jobs=6
completed_jobs=9/9
candidate_rows=10800
```

Smoke:

```text
out=target/pilot_wave/highway_pocket_mutation_001/smoke
seeds=2026,2027,2028
steps=1000
eval_examples=2048
H=192
pockets=4
jackpot=9
jobs=6
completed_jobs=27/27
candidate_rows=243000
```

No raw `target/` outputs are committed.

## Smoke Verdicts

Runner labels:

```text
HIGHWAY_POCKET_MUTATION_POSITIVE
UNGATED_POCKETS_SUFFICIENT
UNRESTRICTED_GRAPH_SUFFICIENT
MUTATION_RESCUES_PHASE_CREDIT_ASSIGNMENT
```

Manual interpretation:

```text
protected highway + pockets: positive smoke
gated writeback specifically: not uniquely proven
unrestricted mutation: still a strong confound
phase-lock bridge: positive micro-bridge, not full phase-lock 004 proof
```

## Symbolic Correction Smoke

| Arm | Final | Heldout | Length | Mention Error | Cancellation | Refocus | Highway Retention | Pocket Ablation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HIGHWAY_ONLY` | `61.9%` | `77.8%` | `42.3%` | `100.0%` | `0.0%` | `100.0%` | `100.0%` | `0.0%` |
| `RANDOM_POCKETS_NO_WRITEBACK` | `61.9%` | `77.8%` | `42.3%` | `100.0%` | `0.0%` | `100.0%` | `100.0%` | `0.0%` |
| `UNGATED_POCKETS` | `90.5%` | `77.8%` | `71.2%` | `0.0%` | `100.0%` | `100.0%` | `100.0%` | `14.3%` |
| `GATED_POCKETS` | `91.4%` | `80.0%` | `77.9%` | `0.0%` | `100.0%` | `100.0%` | `100.0%` | `14.3%` |
| `UNRESTRICTED_GRAPH` | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `100.0%` | `100.0%` | `100.0%` | `20.6%` |

Readout:

```text
The pocket path is not decorative:
  gated pocket ablation max drop = 14.3pp

The protected highway was not damaged:
  highway retention = 100.0%

But gated writeback is not yet uniquely required:
  ungated pockets are close to gated pockets
  unrestricted graph mutation solves the task completely
```

## Phase-Lock Micro Bridge

| Arm | Phase Final | Long Path | Gate Shuffle Control | Highway Phase Retention | Phase Ablation |
|---|---:|---:|---:|---:|---:|
| `HIGHWAY_ONLY_PHASE` | `20.6%` | `20.9%` | `20.6%` | `100.0%` | `0.0%` |
| `RANDOM_POCKETS_NO_WRITEBACK_PHASE` | `20.6%` | `20.9%` | `20.6%` | `100.0%` | `0.0%` |
| `GATED_POCKETS_PHASE` | `100.0%` | `100.0%` | `0.0%` | `100.0%` | `74.3%` |
| `UNRESTRICTED_GRAPH_PHASE` | `100.0%` | `100.0%` | `0.0%` | `100.0%` | `74.4%` |

Readout:

```text
The bridge is nontrivial versus highway-only/random-no-writeback.
The phase pocket ablation is large.
The gate-shuffle control collapses.
```

But this is only a micro-bridge. It does not prove that mutation-selection has solved the full spatial phase-lock credit-assignment failure from 002/003.

## What This Means

Supported:

```text
HIGHWAY_POCKET_MUTATION_POSITIVE as a smoke result.
Sidepockets can evolve useful correction rules while preserving the protected highway.
Pocket contribution is measurable by ablation.
The phase bridge has a clear mutation signal.
```

Not supported:

```text
Gated pockets are uniquely required.
Unrestricted graph mutation is worse.
The full phase-lock spatial credit-assignment problem is solved.
Sidepockets should be promoted into instnct-core public APIs now.
```

## Next Decision

Do not run a broad sidepocket sweep yet.

Best next probe:

```text
STABLE_LOOP_PHASE_LOCK_004_MUTATION_CREDIT_ASSIGNMENT
```

Use the positive micro-bridge as justification, but harden it:

```text
larger spatial phase paths
same-target-neighborhood counterfactuals
distractor gates
heldout path families
gated vs ungated sidepocket separation
unrestricted graph as explicit confound
```

## Claim Boundary

This is a runner-local mutation smoke. It does not prove consciousness, full VRAXION, language grounding, or a production sidepocket architecture.
