# Phase D10 Basin Universality Dossier

Status: scaffolded, not yet a final verdict.

Date: 2026-04-29

## Question

Phase D9.2 and D9.4 established that `seed2042_improved_generalist_v1` is a real H=384 beta.8 generalist research checkpoint and that its current improvement is best explained as an edge-plus-threshold co-adapted package.

Phase D10 asks a stricter question:

```text
Is this a reusable basin/circuit family that returns across seeds, tasks,
and H sizes, or is it a local H=384 seed2042 feature?
```

No "intelligence" or broad scaling claim is allowed from D10 unless the final evidence crosses seed, task, H-size, and causal-diff gates.

## Implemented Scout Modes

`instnct-core/examples/d9_direct_landscape.rs` now has D10 modes for the long-horizon gate:

| mode | purpose |
|---|---|
| `edge-lock-threshold-sweep` | Hold the beta.8 edge substrate fixed and test whether threshold-only local search still has headroom. |
| `threshold-lock-edge-sweep` | Hold the beta.8 threshold timing fixed and test whether edge-only local search still has headroom. |
| `edge-threshold-continued-climb` | Continue local search with the same edge+threshold scope that produced the beta.8 generalist. |
| `scaling-universality-scout` | Run a shared multi-objective scout over one or more checkpoints/H sizes. |
| `task-universality-scout` | Probe whether the same local search mechanism has signal on `echo`, `unigram`, `smooth`, `accuracy`, and the full multi-objective gate. |

All modes write:

```text
candidate_summary.csv
universality_matrix.csv
run_summary.json
causal_summary.json
```

The scout outputs are intentionally compact. Full confirmation still requires 30 fresh eval seeds, longer eval lengths, and causal-diff replay for every promoted winner.

## Verdict Gates

Final D10 verdicts must be assigned conservatively:

| verdict | required evidence |
|---|---|
| `UNIVERSAL_BASIN_CONFIRMED` | At least 3 independent checkpoint seeds, at least 2 H sizes, at least 3 task/gate contexts, 30 fresh-seed confirm, and causal diff for every winner. |
| `LOCAL_H384_BASIN_ONLY` | The beta.8 mechanism remains valid but repeatability fails outside H=384 seed2042. |
| `TASK_SPECIFIC_RESONANCE` | The signal returns only on one task family and fails multi-objective generalization. |
| `SCALING_PROMISING_BUT_INFRA_LIMITED` | Multi-seed or H=512 signal exists, but CPU runtime is insufficient for a final high-H decision. |
| `NO_GENERAL_BASIN` | The beta.8 checkpoint remains useful, but no repeatable family is found. |

## Stop Conditions

- If the D9.4 full causal confirm fails, stop D10 scaling and repair causal localization first.
- If H=384 seed universality fails badly, do not spend GPU time on H=512+.
- If H=512 has no weak signal, do not run H=768 or H=1024.
- If task universality collapses to only one metric, do not claim a general mechanism.

## Current Smoke Status

The initial D10 code smoke was run with low-cost settings (`eval_len=200`, one seed, one climber, one to two steps depending on mode). The goal was schema and loader validation, not scientific inference.

Smoke modes exercised:

```text
edge-lock-threshold-sweep
threshold-lock-edge-sweep
edge-threshold-continued-climb
scaling-universality-scout
task-universality-scout
```

The smokes produced the required D10 schema files. Their rows are too shallow to support a basin verdict.

## Next Full Runs

1. Run D9.4b full causal confirm on `seed2042_improved_generalist_v1`.
2. If D9.4b holds, run the three geometry modes at `eval_len=4000` with 30 fresh seeds.
3. Run H=384 seed universality across at least 10 checkpoints.
4. Run task universality ladder.
5. Only promote H=512/GPU work if the H=384 multi-seed gate is at least weak positive.

## Interpretation Guardrail

D10 uses the current Rust core's `H`/genome dimensionality as the working proxy for "D". A separate GPU VRAXION model path would need its own validation boundary and must not inherit D10 claims automatically.
