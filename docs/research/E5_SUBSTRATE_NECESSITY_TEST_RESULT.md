# E5_SUBSTRATE_NECESSITY_TEST Result

Status: complete.

Canonical runner:

```text
scripts/probes/run_e5_substrate_necessity_test.py
```

Canonical checker:

```text
scripts/probes/run_e5_substrate_necessity_test_check.py
```

Canonical artifact root:

```text
target/pilot_wave/e5_substrate_necessity_test_parallel_strict
```

## Decision

```text
decision = e5_leak_or_artifact_detected
winner = e4_top_down_hierarchical_router
next = E5L_REPAIR_SUBSTRATE_TASK_AND_LEAK_CONTROLS
checker_failure_count = 0
deterministic_replay_passed = true
```

## What Ran

Balanced evidence configuration:

```text
seeds = 76001,76002,76003,76004,76005
train/validation/heldout/OOD/counterfactual/adversarial rows per seed = 800/300/300/300/300/300
gradient_epochs = 120
mutation_generations = 80
population_size = 24
execution_mode = parallel
parallel_workers = 9
```

Systems:

```text
e4_top_down_hierarchical_router
tiny_mlp_gradient
tiny_mlp_mutation_only
tiny_recurrent_gradient
tiny_recurrent_mutation_only
hybrid_neural_frontend_mutation_router
flat_detail_scanner
bottom_up_evidence_scanner
random_classifier
oracle_reference_only
```

## Key Metrics

```text
e4_top_down_hierarchical_router:
  heldout/OOD/counterfactual/adversarial usefulness = 1.0 / 1.0 / 1.0 / 1.0
  level/path/stop accuracy = 1.0 / 1.0 / 1.0
  over-detail / irrelevant-branch = 0.0 / 0.0
  branch-order shuffled usefulness = 1.0
  pass with leak controls = true

tiny_mlp_gradient:
  heldout/OOD/counterfactual/adversarial usefulness = 1.0 / 1.0 / 1.0 / 1.0
  branch-order shuffled usefulness = 0.21564
  pass with leak controls = false

tiny_recurrent_gradient:
  heldout/OOD/counterfactual/adversarial usefulness = 1.0 / 1.0 / 1.0 / 0.99838
  branch-order shuffled usefulness = 0.484266666667
  pass with leak controls = false

tiny_mlp_mutation_only:
  heldout usefulness = 0.16192
  branch-order shuffled usefulness = 0.055546666667
  pass = false

tiny_recurrent_mutation_only:
  heldout usefulness = 0.232246666667
  branch-order shuffled usefulness = 0.09926
  pass = false

hybrid_neural_frontend_mutation_router:
  heldout usefulness = 0.19
  branch-order shuffled usefulness = 0.19
  pass = false

bottom_up_evidence_scanner:
  heldout usefulness = 0.896066666667
  adversarial usefulness = 0.692146666667
  over-detail / irrelevant-branch = 0.146 / 0.059333333333
  pass = false

flat_detail_scanner:
  heldout usefulness = 0.0
  pass = false
```

## Interpretation

E5 does not support the claim that neural nets are necessary for the current controlled abstraction-routing proxy. The non-neural E4 top-down router remains the only clean passing system and is robust to branch-order shuffle.

However, E5 also does not cleanly validate gradient neural abstraction routing. The gradient MLP and recurrent models reached perfect normal heldout/OOD/counterfactual metrics, but failed the branch-order remap control. That means their success is consistent with fixed candidate-position/order shortcut learning rather than order-invariant abstraction-level routing.

Mutation-only neural did not work in this setup. The hybrid system also did not beat or match the non-neural router.

## Boundary

This remains a controlled symbolic substrate necessity test. It does not prove AGI, consciousness, natural-language reasoning, or model-scale behavior.
