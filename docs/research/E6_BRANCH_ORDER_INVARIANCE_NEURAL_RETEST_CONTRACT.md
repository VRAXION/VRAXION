# E6 Branch-Order Invariance Neural Retest Contract

## Purpose

`E6_BRANCH_ORDER_INVARIANCE_NEURAL_RETEST` retests the E5 neural shortcut result under explicit branch-order randomization.

E5 found that gradient neural models could score well on the normal E4 abstraction-routing proxy while failing branch-order shuffle controls. E6 asks whether that failure is fixed by randomized-order training, or whether a neural architecture with choicewise/shared scoring is required.

## Boundary

E6 is a controlled symbolic proxy test. It does not prove natural-language reasoning, AGI, consciousness, or model-scale behavior.

## Reused Task

E6 imports the E5 runner and reuses the E4/E5 symbolic row encoding and evaluation semantics:

- heads: `level`, `verdict`, `descend`, `cause`, `mechanism`, `evidence`, `stop_depth`
- max choices per head: `8`
- feature dim: `14`
- fixed padded input: `7 x 8 x 14`
- splits: train, validation, heldout, OOD, counterfactual, adversarial

## Systems

- `e4_top_down_reference`: non-neural top-down mutation/rollback reference.
- `mlp_fixed_order_gradient`: E5-style MLP negative control.
- `mlp_random_order_gradient`: same MLP, but choices are randomly permuted during training with targets remapped.
- `recurrent_fixed_order_gradient`: E5-style recurrent negative control.
- `recurrent_random_order_gradient`: same recurrent model, randomized-order training.
- `choicewise_shared_random_order_gradient`: shared per-choice scorer with randomized-order training.
- `random_classifier`: sanity control.

## Long-Run Requirements

- The run must write `progress.jsonl` at startup, per system, every epoch, every mutation generation, and final artifact write.
- Gradient systems must write `e6_training_history_<system>.json` every epoch.
- The top-down mutation reference must write mutation history every generation.
- The runner must support parallel system execution so CPU mutation and GPU neural training can overlap.
- Deterministic replay must hash-match required artifacts.

## Required Artifacts

- `e6_backend_manifest.json`
- `e6_task_generation_report.json`
- `e6_invariance_comparison_report.json`
- `e6_branch_order_report.json`
- `e6_deterministic_replay_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`
- `progress.jsonl`
- per-system candidate summaries and parameter diffs
- per-gradient-system training histories
- top-down mutation history
- row-level samples for heldout, OOD, counterfactual, and adversarial splits

## Decision Rules

- `e6_branch_order_invariance_training_succeeds`: randomized-order MLP or recurrent neural model passes normal and branch-order controls.
- `e6_order_equivariant_neural_architecture_viable`: the choicewise shared scorer passes normal and branch-order controls when ordinary randomized-order MLP/RNN do not.
- `e6_non_neural_router_remains_preferred`: only the top-down non-neural reference cleanly passes.
- `e6_leak_or_replay_failure`: label shuffle control or deterministic replay fails.
- `e6_invariance_retest_inconclusive`: none of the above is clean.

## Pass Threshold

A system passes clean invariant routing only if heldout/OOD/counterfactual/adversarial usefulness is at least `0.95`, level/path/stop accuracy is at least `0.95`, over-detail is at most `0.03`, irrelevant branch expansion is at most `0.03`, and the same routing gates pass under branch-order shuffle.
