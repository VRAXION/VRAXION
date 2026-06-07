# E8B Producer Write-Code Dynamics And Loss Audit Result

## Status

```text
primary_decision = e8b_producer_capacity_or_loss_bottleneck
gradient_diagnostic_decision = e8b_gradient_conflict_or_jagged_target_confirmed
primary_checker_failure_count = 0
gpu_confirm_checker_failure_count = 0
grad_diagnostic_checker_failure_count = 0
gpu_grad_diagnostic_checker_failure_count = 0
deterministic_replay = pass on all checked roots
```

E8B confirms that the producer write-code bottleneck is not a simple
end-to-end training crash. The high-batch primary run shows smooth learning and
early plateau. A targeted low-batch gradient diagnostic then shows real
batch-to-batch gradient direction conflict, so the better interpretation is:

```text
producer plateau = capacity/loss/teacher-code bottleneck
with additional jagged/conflicting-gradient evidence when measured at
smaller diagnostic batch size.
```

## Artifact Roots

```text
primary:
target/pilot_wave/e8b_producer_write_code_dynamics_and_loss_audit/

gpu_confirm:
target/pilot_wave/e8b_producer_write_code_dynamics_and_loss_audit_gpu_confirm/

cpu_gradient_diagnostic:
target/pilot_wave/e8b_producer_write_code_dynamics_and_loss_audit_grad_diagnostic/

gpu_gradient_diagnostic:
target/pilot_wave/e8b_producer_write_code_dynamics_and_loss_audit_gpu_grad_diagnostic/
```

## Primary Evidence Run

Configuration:

```text
seeds = 102001..102008
device = cpu
cpu_workers = 8
batch_size = 160
rows = train 180/seed, validation/heldout/OOD/counterfactual/adversarial 96/seed
checker = pass
deterministic_replay = pass
```

Primary aggregate:

```text
decision = e8b_producer_capacity_or_loss_bottleneck
best_system = local_smooth_teacher_code
best_usefulness = 0.448403279770
oracle_low_bit_reference = 0.616471354167
consumer_distill_reference = 0.542693014706
current_projection_teacher_baseline = 0.431529606144
```

Key primary rows:

| system | usefulness | code_sim | val_code_sim_best | tail_gain | tail_range | grad_norm |
|---|---:|---:|---:|---:|---:|---:|
| current_projection_teacher_baseline | 0.431530 | 0.697548 | 0.693419 | 0.010631 | 0.010648 | 0.281358 |
| simplified_canonical_teacher | 0.445868 | 0.845864 | 0.829999 | 0.003228 | 0.003343 | 0.236000 |
| local_smooth_teacher_code | 0.448403 | 0.874774 | 0.850900 | 0.003300 | 0.003594 | 0.231262 |
| per_cell_supervised_loss | 0.382482 | 0.714693 | 0.727313 | 0.015204 | 0.015204 | 0.368663 |
| code_similarity_loss | 0.380027 | 0.707304 | 0.718087 | 0.012042 | 0.012042 | 1.899473 |
| mutation_only_from_random_lowbit | 0.441817 | 0.631454 | n/a | n/a | n/a | n/a |
| mutation_repair_after_similarity_threshold | 0.396544 | 0.715029 | 0.727313 | 0.015204 | 0.015204 | 0.368663 |

The primary run shows:

```text
loss/code-sim improves smoothly
validation tail_gain is small
local_smooth/simplified teachers improve code similarity
but usefulness remains far below oracle/consumer reference
mutation repair after producer training does not close the gap
```

Important caveat:

```text
primary gradient_variance_mean = 0.0
primary gradient_cosine_mean = 0.0
```

That was not treated as proof that gradients are non-conflicting. With
`batch_size=160`, the per-skill diagnostic often degenerates into too few
batches for meaningful cosine/variance measurement.

## GPU Confirm

Configuration:

```text
seeds = 102009..102012
device = cuda
batch_size = 128
checker = pass
deterministic_replay = pass
```

Result:

```text
decision = e8b_producer_capacity_or_loss_bottleneck
best_system = mutation_only_from_random_lowbit
best_usefulness = 0.442795249042
oracle_low_bit_reference = 0.621679687500
consumer_distill_reference = 0.485195530726
gradient_conflict_flag = false
```

The GPU confirm agreed with the primary decision class. The winner changed in
the smaller confirm run, but the system still stayed far below oracle and did
not make producer write-code learning viable.

## Gradient Diagnostic

Because the primary gradient diagnostics were degenerate, a smaller-batch audit
was run:

```text
cpu_grad_diagnostic:
  seeds = 102013..102016
  batch_size = 24
  checker = pass
  deterministic_replay = pass

gpu_grad_diagnostic:
  seeds = 102017..102018
  batch_size = 24
  checker = pass
  deterministic_replay = pass
```

CPU gradient diagnostic:

```text
decision = e8b_gradient_conflict_or_jagged_target_confirmed
best_system = mutation_only_from_random_lowbit
best_usefulness = 0.463549202241
oracle_low_bit_reference = 0.636979166667
consumer_distill_reference = 0.655284926471
```

GPU gradient diagnostic:

```text
decision = e8b_gradient_conflict_or_jagged_target_confirmed
best_system = codebook_teacher
best_usefulness = 0.480071478939
oracle_low_bit_reference = 0.607812500000
consumer_distill_reference = 0.642222222222
```

Representative gradient rows:

| run | system | usefulness | code_sim | val_best | grad_var | grad_cos | grad_neg_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu_grad | current_projection_teacher_baseline | 0.453718 | 0.827424 | 0.765807 | 0.021805 | 0.037815 | 0.470486 |
| cpu_grad | local_smooth_teacher_code | 0.460634 | 0.916290 | 0.853183 | 0.020397 | -0.047867 | 0.539435 |
| cpu_grad | codebook_teacher | 0.445310 | 0.802728 | 0.738042 | 0.022748 | 0.015410 | 0.500000 |
| cpu_grad | per_cell_supervised_loss | 0.351140 | 0.830857 | 0.829158 | 0.008883 | 0.440574 | 0.177331 |
| gpu_grad | current_projection_teacher_baseline | 0.479522 | 0.803488 | 0.753046 | 0.027483 | 0.057279 | 0.425347 |
| gpu_grad | local_smooth_teacher_code | 0.437773 | 0.906525 | 0.852089 | 0.026028 | -0.046916 | 0.536458 |
| gpu_grad | codebook_teacher | 0.480071 | 0.785537 | 0.736372 | 0.023908 | 0.074782 | 0.438079 |

This answers the original diagnostic concern directly:

```text
It was not wild metric oscillation at the output level.
It was smooth plateau plus batch-level gradient disagreement.
```

The strongest version is visible in `local_smooth_teacher_code`:

```text
primary code_sim high, tail_gain tiny
small-batch grad_cos below zero
negative gradient-cosine rate around 0.54
```

So local smoothing makes the code easier to imitate, but does not make the
downstream composition problem solved.

## What E8B Proved

```text
1. Producer writes do learn part of the teacher code.
2. The learning curve is mostly smooth, not exploding or chaotic.
3. Validation improvement plateaus early.
4. Better teachers improve code similarity, especially local_smooth and simplified.
5. Higher code similarity alone does not close downstream usefulness.
6. Mutation repair after similarity threshold did not rescue the producer.
7. Mutation-only from random low-bit can be competitive in some small runs,
   but it did not prove producer write-code learning viability.
8. Small-batch diagnostics expose real gradient direction conflict.
```

## What E8B Did Not Prove

```text
1. It did not prove the current producer architecture is sufficient.
2. It did not prove a single output-code teacher is the right target.
3. It did not prove mutation-only producer learning is viable.
4. It did not prove oracle-like composition without oracle/reference writes.
5. It did not test image, language, or model-scale behavior.
```

## Interpretation

The producer bottleneck is now more concrete:

```text
not just "bad calibration"
not just "single metric artifact"
not just "wild oscillation"

but:
teacher-code imitation improves,
then plateaus below useful composition,
and small-batch gradients show conflicting directions.
```

The likely failure is a combination of:

```text
teacher-code / producer-loss mismatch
downstream consumer compatibility not captured by raw code_sim
target format still too jagged or overconstrained for one producer head
```

## Recommended Next Step

Run:

```text
E8C_PRODUCER_TARGET_DECOMPOSITION_AND_CONSUMER_COMPATIBILITY_PROBE
```

Core question:

```text
Can the producer write target be decomposed into smaller, lower-conflict
subtargets that preserve downstream consumer compatibility better than raw
teacher-code imitation?
```

Suggested comparisons:

```text
1. current full-code teacher
2. per-skill decomposed teacher heads
3. staged primary-bit then support-bits teacher
4. consumer-compatibility-weighted code targets
5. route-step-local teacher targets
6. mutation repair only after consumer-compatible plateau
```

Decision focus:

```text
If decomposed targets reduce gradient negative rate and improve usefulness,
the E8 bottleneck is target/loss decomposition.

If gradient conflict drops but usefulness stays low,
the bottleneck is producer architecture or consumer interface.

If nothing improves, the current canonical RAM-code interface is still wrong.
```
