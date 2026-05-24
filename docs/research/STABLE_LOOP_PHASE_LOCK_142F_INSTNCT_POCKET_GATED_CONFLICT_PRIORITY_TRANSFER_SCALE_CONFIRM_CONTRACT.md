# STABLE_LOOP_PHASE_LOCK_142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM

## Contract

142F is an executable helper-only scale confirm after the positive 142A conflict/priority smoke probe. It confirms scaled conflict-priority final selection under constrained helper/backend evidence only.

It may call `shared_raw_generation_helper.raw_generate` from the runner only, through the shared helper, and only with these request keys: `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`. The checker must not call `raw_generate`.

## Upstream

142F requires 142A:

```text
decision = instnct_pocket_gated_conflict_priority_transfer_probe_positive
next = 142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM
main_final_answer_accuracy = 1.0
priority_rule_accuracy = 1.0
conflict_resolution_accuracy = 1.0
priority_inversion_accuracy = 1.0
same_template_opposite_winner_accuracy = 1.0
main_pocket_writeback_rate = 1.0
ablation_final_answer_accuracy = 0.0
pocket_ablation_delta_final_answer_accuracy = 1.0
shortcut and bypass rates = 0.0
deterministic_replay_passed = true
```

## Scale Gates

142F scales to 4 seeds, 6 families, 24 groups per family, group size 5, 2880 eval rows, and at least 72 scaffold variants.

Positive requires:

```text
main_final_answer_accuracy >= 0.90
priority_rule_accuracy >= 0.90
conflict_resolution_accuracy >= 0.90
priority_inversion_pair_count >= declared floor
priority_inversion_accuracy >= 0.85
same_template_opposite_winner_accuracy >= 0.85
main_pocket_writeback_rate >= 0.95
main_contrast_group_accuracy >= 0.90
ablation_final_answer_accuracy <= 0.05
pocket_ablation_delta_final_answer_accuracy >= 0.85
wrong_priority_field_rate = 0.0
priority_default_shortcut_rate = 0.0
always_a_shortcut_rate = 0.0
always_b_shortcut_rate = 0.0
table_default_shortcut_rate = 0.0
rule_default_shortcut_rate = 0.0
visible/noisy/direct marker rates = 0.0
deterministic_replay_passed = true
```

The winner distribution is a hard gate: A, B, table, and rule winner rates must each be at least 0.15. Per-winner, per-seed, and per-family gates must independently pass.

Required hardening artifacts include `helper_request_audit.json`, `canonical_metric_alias_report.json`, `per_seed_gate_report.json`, `per_family_gate_report.json`, `winner_distribution_report.json`, `per_winner_gate_report.json`, `priority_inversion_pair_report.json`, `same_template_opposite_winner_report.json`, and `shortcut_report.json`.

## Decision

Positive route:

```text
decision = instnct_pocket_gated_conflict_priority_transfer_scale_confirmed
verdict = INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRMED
next = 142Z_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_NEXT_DECISION_PLAN
```

Clean negative routes include wrong priority selection, priority default shortcut, priority inversion failure, conflict resolution scale failure, pocket ablation non-criticality, and helper integrity failure.

## Boundary

This confirms constrained helper/backend conflict-priority final selection only. It is not open-ended reasoning, not general composition, not GPT-like readiness, not open-domain reasoning, not broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.
