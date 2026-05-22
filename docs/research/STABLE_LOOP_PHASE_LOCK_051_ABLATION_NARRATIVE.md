# STABLE_LOOP_PHASE_LOCK_051 Ablation Narrative

## What 049/050 Showed

The passing 049 arm used the route-grammar train-and-infer path on the
adversarial frozen corpus. The 050 audit reran that result fresh, checked source
hashes, validated metric schema, checked leakage, and documented known failure
controls.

The main positive reproduced by 050 was:

```text
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER:
  heldout_exact_accuracy = 1.000
  ood_exact_accuracy = 1.000
  family_min_accuracy = 1.000
  hard_distractor_accuracy = 1.000
  long_ood_accuracy = 1.000
  unique_output_count = 75 / 75
  collapse_detected = false
```

## Why The Controls Matter

The no-route baseline collapses to one output and family-min stays zero. The
048 reference collapses under the larger adversarial frozen scale. Shuffled
labels destroy exact accuracy. Random labels and random phase rules fail
family-min or long-OOD gates. Always-space, always-majority, and copy-last
controls directly trigger the shortcut/collapse checks.

This means the paper table is not only reporting a high aggregate score. It
also records the failure modes that would catch static output, majority label,
copy shortcut, random label, and random phase false positives.

## Claim Boundary

Supports:

```text
reviewer-facing reproduction package for bounded 049/050 adversarial frozen-eval result
```

Does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```
