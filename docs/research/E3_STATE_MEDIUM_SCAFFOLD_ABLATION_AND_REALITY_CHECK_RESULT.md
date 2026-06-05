# E3_STATE_MEDIUM_SCAFFOLD_ABLATION_AND_REALITY_CHECK Result

## Outcome

```text
decision = e3_recurrence_not_required
supporting_decision_flags = e3_recurrence_not_required,e3_handcrafted_projector_dependency_detected
next = E4_ISOLATE_NONRECURRENT_NONLINEAR_PROJECTION_ADVANTAGE
checker_failure_count = 0
```

## Output

```text
final_evidence_path = target/pilot_wave/e3_state_medium_scaffold_ablation_and_reality_check_rescaled/
partial_overscale_path = target/pilot_wave/e3_state_medium_scaffold_ablation_and_reality_check/
smoke_path = target/pilot_wave/e3_smoke/
```

## Scale

```text
seeds = 74001,74002,74003,74004,74005
train_rows = 2000
validation_rows = 750
heldout_rows = 750
ood_rows = 750
counterfactual_rows = 750
adversarial_rows = 750
population_size = 16
generations = 40
mutation_attempts_per_variant = 640
accepted_mutations_total = 1428
rejected_mutations_total = 4332
rollback_count_total = 4332
```

## Variant Results

```text
e2_original_projector: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=4.0445414485
random_projector: pass=False heldout=0.0 ood=0.0 counterfactual=0.0 adversarial=0.0 gap=-0.166708361135
zero_projector: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=5.707037714939
permuted_projector: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=2.331319843816
dense_random_projector: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=1.951311784234
no_recurrence: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=3.84861602763
final_state_only: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=4.252011579939
nonrecurrent_nonlinear: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=0.552671316029
stronger_flat_pairwise: pass=True heldout=1.0 ood=1.0 counterfactual=1.0 adversarial=1.0 gap=1.431098304926
```

## Interpretation

```text
E2 does not reduce purely to the exact hand-crafted projector: zero, permuted, and dense-random projector variants passed.
However, sparse random_projector failed, so scaffold removal was not uniformly robust.
Recurrence was not required on this task: no_recurrence passed close enough to the original projector.
Trajectory readout was not proven necessary: final_state_only matched or exceeded the original heldout gap.
The stronger flat/pairwise baseline passed the task, but did not match the original gap threshold used by the E3 decision rule.
```

## Controls

```text
leakage_sentinel_passed = True
route_index_leak_detected = False
candidate_name_leak_detected = False
shuffled_route_order_passed = True
synthetic_harness_only = False
static_metric_dictionary_used = False
hardcoded_improvement_used = False
gradient_backprop_used = False
row_level_predictions_used = True
internal_replay_passed = True
deterministic_replay_passed = True
```

## Boundary

E3 is a real-backend state-medium scaffold ablation on the E2 task. It does not prove model-scale reasoning, AGI, consciousness, production readiness, natural-language training, tokenizer learning, or Raven-scale capability.
