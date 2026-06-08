# E07 Binary Flow Matrix Pocket Scheduling Confirm Result

Status: completed.

## Decision

```text
decision = e07_binary_flow_matrix_pocket_scheduling_confirmed
next = E08_COMMON_MATRIX_LANGUAGE_TRANSLATION_CONFIRM
best_default_triggered_gated_arm = DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e07_binary_flow_matrix_pocket_scheduling_confirm/
```

## Search Result

No equivalent pre-existing E-line probe was found after fetching the current
remote refs and searching local/remote branches.

Closest related files:

```text
docs/research/E7B_POCKET_ROUTING_COMPOSITION_PROBE_CONTRACT.md
docs/research/E7E_FLOW_PIPE_DRIFT_AND_ROUTER_REPAIR_PROBE_CONTRACT.md
docs/research/E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE_CONTRACT.md
```

Summary:

```text
E7B: symbolic pocket routing over frozen pockets.
E7E: flow-pipe drift, route-around, and local repair.
E8F: proposal memory and router commit in a numeric pocket-router proxy.
```

Those files do not jointly test binary Flow/Main matrix state, common
matrix-language proposal fields, always-on cheap transforms versus triggered
complex pockets, snapshot versus rollout pocket selection, and branch
contamination/destructive-overwrite gates in a stdlib-only runner.

## Key Metrics

| arm | state accuracy | useful recall | wrong commit | destructive overwrite | branch contam | cost/tick |
|---|---:|---:|---:|---:|---:|---:|
| SNAPSHOT_SELECTED_POCKET | 0.812 | 0.818 | 0.287 | 0.187 | 0.000 | 2.400 |
| ROLLOUT_SELECTED_POCKET | 0.987 | 0.988 | 0.011 | 0.008 | 0.000 | 3.000 |
| ALL_COMPLEX_ALWAYS_NO_GATE | 0.579 | 0.955 | 0.749 | 0.338 | 0.335 | 12.000 |
| ALL_COMPLEX_ALWAYS_GATED | 0.594 | 0.988 | 0.615 | 0.378 | 0.000 | 12.400 |
| DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED | 0.987 | 0.988 | 0.011 | 0.008 | 0.000 | 1.856 |
| DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED | 0.987 | 0.988 | 0.011 | 0.008 | 0.000 | 1.857 |

Positive-gate deltas for the best default arm versus
`ALL_COMPLEX_ALWAYS_NO_GATE`:

```text
final_state_accuracy_delta = +0.408036
wrong_commit_rate_reduction = 0.985650
destructive_overwrite_rate_reduction = 0.976855
branch_contamination_rate_reduction = 1.000000
avg_cost_per_tick_reduction = 0.845317
false_positive_rate_delta = -0.737835
```

## Confirmed Findings

Snapshot-only selection picked `345` snapshot-trap pockets and failed the
temporal comparison. Rollout selection reduced temporal drift from `0.188442`
to `0.012897` and reduced wrong commit rate from `0.287415` to `0.010742`.

The common matrix-language contract passed. Gated arms rejected direct local
pocket dialect output before commit, while the no-gate baseline committed
`1413` direct dialect mutations and recorded a branch contamination rate of
`0.335152`.

The default triggered gated arm preserved useful update recall at `0.988293`
while reducing complex calls from `6.0` per tick to `0.952381` per tick.

## Verification

```text
python3 scripts/probes/run_e07_binary_flow_matrix_pocket_scheduling_confirm.py
python3 scripts/probes/run_e07_binary_flow_matrix_pocket_scheduling_confirm_check.py --out target/pilot_wave/e07_binary_flow_matrix_pocket_scheduling_confirm --write-summary
```

The checker passed with `failure_count = 0`.

Boundary: E07 is a deterministic synthetic binary-matrix scheduling probe only.
It does not claim real-world deployment readiness, model-scale capability, or
broad reasoning behavior.
