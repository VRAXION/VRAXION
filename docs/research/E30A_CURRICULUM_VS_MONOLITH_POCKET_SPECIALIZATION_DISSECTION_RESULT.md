# E30A Curriculum vs Monolith Pocket Specialization Dissection Result

Status: complete.

## Decision

```text
decision = e30a_monolith_sufficient
checker_failure_count = 0
target_checker_passed = true
sample_only_checker_passed = true
```

## Setup

E30A compared the same Flow/Pocket architecture under different training paths:

```text
monolith_direct_final
curriculum_staged_final
random_order_curriculum_control
reverse_curriculum_control
random_static_control
```

The task was controlled naturalized text, not FineWeb. The point was introspection: Pocket Operator specialization, Arbiter behavior, Trace Ledger validity, and ablation locality.

## Key Results

```text
monolith_direct_final:
  heldout_resolution = 0.828571
  trap_resolution = 0.890476
  phrase_holdout_resolution = 0.821429
  heldout_action_accuracy = 0.897619
  heldout_trace_exact = 0.326190
  heldout_trace_bit_accuracy = 0.853571
  pocket_specialization = 0.449069
  ablation_locality = 0.076365
  wrong_confident_on_unresolved = 0.000000

curriculum_staged_final:
  heldout_resolution = 0.635714
  trap_resolution = 0.685714
  phrase_holdout_resolution = 0.645238
  heldout_action_accuracy = 0.635714
  heldout_trace_exact = 0.000000
  heldout_trace_bit_accuracy = 0.759821
  pocket_specialization = 0.291115
  ablation_locality = 0.000000
  wrong_confident_on_unresolved = 1.000000

random_order_curriculum_control:
  heldout_resolution = 0.635714
  wrong_confident_on_unresolved = 0.713533

reverse_curriculum_control:
  heldout_resolution = 0.635714
  wrong_confident_on_unresolved = 1.000000
```

## Interpretation

The expected curriculum-positive result did not appear. Direct final-task training was stronger and cleaner on this controlled task. It also had higher Pocket Operator specialization and nonzero ablation locality, while the staged curriculum variants remained diffuse and produced high wrong-confident unresolved behavior.

This is a useful negative result:

```text
naive staged curriculum != automatic Pocket Operator specialization
```

The likely failure mode is curriculum interference. Stage5 unresolved training and earlier simplified stages biased the model into a poorer final-task basin. The monolith direct model saw the full mixed distribution from the start and learned a more useful internal partition.

## Next Step

Do not scale E30A as-is. The next clean test should change the curriculum design, not just increase data:

```text
E30B_INTERLEAVED_CURRICULUM_AND_DISTILLATION_DISSECTION
```

Candidate repairs:

```text
interleave earlier stages with final mixed examples
use replay buffer so stage5 does not overwrite answerable behavior
distill monolith final behavior into curriculum checkpoints
add explicit unresolved contrastive pairs during every stage
measure Pocket Operator specialization after each stage
```

## Boundary

E30A is a controlled naturalized-text dissection probe. It is not a chatbot, production system, raw language reasoning proof, AGI claim, consciousness claim, or model-scale claim.
