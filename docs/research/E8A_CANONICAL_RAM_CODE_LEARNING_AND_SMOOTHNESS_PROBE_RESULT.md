# E8A Canonical RAM Code Learning And Smoothness Probe Result

## Status

```text
primary_decision = e8a_mutation_repair_after_distillation_positive
primary_best_system = consumer_distill_binary
primary_checker_failure_count = 0
primary_deterministic_replay_passed = true

gpu_confirm_decision = e8a_producer_write_bottleneck
gpu_confirm_best_system = consumer_distill_binary
gpu_confirm_checker_failure_count = 0
gpu_confirm_deterministic_replay_passed = true
```

Artifact roots:

```text
target/pilot_wave/e8a_canonical_ram_code_learning_and_smoothness_probe/
target/pilot_wave/e8a_canonical_ram_code_learning_and_smoothness_probe_gpu_confirm/
```

## Primary Evidence

Primary run used 8 deterministic seeds:

```text
101001, 101002, 101003, 101004, 101005, 101006, 101007, 101008
```

Mean usefulness:

```text
oracle_low_bit_reference                    0.621354
consumer_distill_binary                     0.570588
contrastive_ram_code_alignment              0.454278
mutation_repair_after_distillation          0.444620
dense_graph_danger_control                  0.435837
full_end_to_end_control                     0.435837
progressive_code_freeze                     0.426951
current_best_baseline                       0.426951
producer_consumer_staged_int4               0.424066
producer_distill_int4                       0.424066
mutation_only_from_random_lowbit            0.417963
soft_to_hard_int4_to_ternary_to_binary      0.409411
producer_consumer_staged_binary             0.396498
producer_distill_binary                     0.396498
producer_consumer_staged_ternary            0.385780
producer_distill_ternary                    0.385780
```

Interpretation:

```text
consumer_distill_binary reads canonical low-bit RAM code well when the oracle code
is supplied as input, but learned producer/staged systems do not yet write that code
cleanly enough.

mutation_repair_after_distillation improves over the int4 producer/staged branch
and over the current baseline, but it does not close the gap to the oracle
low-bit reference.

mutation_only_from_random_lowbit did not win.
```

## GPU Confirm

GPU confirm used seeds:

```text
101009, 101010
```

Mean usefulness:

```text
oracle_low_bit_reference                    0.603125
consumer_distill_binary                     0.563889
mutation_repair_after_distillation          0.472212
producer_consumer_staged_int4               0.466041
producer_distill_int4                       0.466041
current_best_baseline                       0.450385
contrastive_ram_code_alignment              0.443621
soft_to_hard_int4_to_ternary_to_binary      0.424882
dense_graph_danger_control                  0.423427
full_end_to_end_control                     0.423427
mutation_only_from_random_lowbit            0.422309
producer_consumer_staged_ternary            0.421855
producer_distill_ternary                    0.421855
producer_consumer_staged_binary             0.417049
producer_distill_binary                     0.417049
```

The GPU confirm agrees with the main bottleneck: the consumer can use canonical
binary RAM writes, but producer-side code learning remains weak.

## Scientific Read

E8A supports this narrower claim:

```text
canonical low-bit RAM code is useful and readable,
but final-answer pressure plus direct producer distillation is not enough
to reliably make pockets write the canonical code.
```

It does not support a claim that mutation from random low-bit code can discover
the RAM language. Mutation repair helped only after a trained/distilled starting
point and stayed below the oracle reference.

## Training Dynamics Sanity

E8A did not log raw gradient norms. It did log per-epoch loss and validation
accuracy for every masked producer/distillation training run. Those curves show
mostly smooth loss descent with early validation plateau, not chaotic oscillation.

Mean over 48 producer groups per system:

```text
system                                 loss_start loss_end  loss_drop val_start val_end  val_best tail_gain tail_range
producer_distill_binary                0.857041   0.724714  0.132327  0.678819  0.703646 0.743924 0.002865  0.016580
producer_distill_ternary               0.790204   0.668884  0.121320  0.660503  0.696007 0.740799 0.001476  0.013628
producer_distill_int4                  0.655322   0.573666  0.081656  0.676997  0.708420 0.751215 0.005035  0.013628
contrastive_ram_code_alignment         0.626672   0.560064  0.066608  0.694184  0.707378 0.755642 0.004514  0.015451
soft_to_hard_int4_to_ternary_to_binary 0.759512   0.655823  0.103689  0.723785  0.716840 0.754080 0.001823  0.013542
```

Interpretation:

```text
loss kept decreasing
validation mostly plateaued by the middle epochs
tail oscillation was small
best validation usually happened around epoch 5-7
final validation often sat below best validation by roughly 0.04
```

So the failure was not "optimizer exploded" or "training oscillated wildly".
The clearer signal is representational: final producer writes still diverged
from the oracle RAM code.

Mean evaluation write-code mismatch:

```text
system                         usefulness code_similarity bundle_mae next_pocket_error sign_mismatch
oracle_low_bit_reference       0.621354   1.000000        0.000000   0.000000          0.000000
consumer_distill_binary        0.570588   1.000000        0.000000   0.000000          0.000000
producer_distill_binary        0.396498   0.294278        0.705722   0.289597          0.371624
producer_distill_ternary       0.385780   0.326501        0.673499   0.276918          0.368491
producer_distill_int4          0.424066   0.669019        0.330981   0.135823          0.357528
contrastive_ram_code_alignment 0.454278   0.831727        0.168273   0.062003          0.290282
mutation_repair_after_distill  0.444620   0.669045        0.330955   0.135983          0.359993
```

This is why the stricter read is:

```text
consumer can read canonical RAM code
producer does not reliably write canonical RAM code
contrastive/simplified alignment improves code similarity but still does not
reach the oracle reference
```

Worst producer skills by sampled step MAE:

```text
producer_distill_binary:
  parity                mean_mae=0.7422
  counterfactual_flip   mean_mae=0.6562
  threshold             mean_mae=0.6380

producer_distill_int4:
  counterfactual_flip   mean_mae=0.3986
  parity                mean_mae=0.3818
  threshold             mean_mae=0.3497

contrastive_ram_code_alignment:
  counterfactual_flip   mean_mae=0.2219
  threshold             mean_mae=0.1736
  parity                mean_mae=0.1500
```

This suggests the next probe should log raw gradient norm, code-similarity per
epoch, per-cell entropy, and per-skill code MAE during training, not only final
evaluation.

## Next Step

The next falsification target should isolate producer write learning:

```text
E8B_PRODUCER_CANONICAL_CODE_SHAPING_PROBE
```

Recommended focus:

```text
1. producer-only supervised RAM-code writing with stronger per-cell losses
2. simplified canonical teacher vs current projection teacher
3. contrastive/codebook teacher variants
4. write-code curriculum before composition
5. mutation repair after producer code similarity crosses a threshold
```

## Boundary

E8A is a controlled numeric pocket-router proxy. Oracle writes were allowed only
as teacher targets or diagnostic references. Learned systems were evaluated
without oracle writes at inference. This result makes no raw-language, AGI,
consciousness, deployed-model, or model-scale claim.
