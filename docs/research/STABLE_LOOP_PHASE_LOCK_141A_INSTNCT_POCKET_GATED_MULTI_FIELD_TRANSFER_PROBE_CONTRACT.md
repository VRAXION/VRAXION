# STABLE_LOOP_PHASE_LOCK_141A_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_PROBE

141A is the executable helper-only follow-up selected by 140Z after the
positive 140Y multi-step transfer scale confirm.

The probe asks whether the pocket-gated path can select a final answer that is
not any single visible or intermediate field. Each row contains field A, field
B, an intermediate value, optional table/rule fields, visible wrong values, and
noisy distractors. The correct output is the machine-readable derived final
field carried through the open pocket markers.

Required families:

- FIELD_A_PLUS_FIELD_B_TO_FINAL
- POCKET_SOURCE_TABLE_RULE_FIELD
- DUAL_POCKET_PRIORITY_CONFLICT
- MULTI_FIELD_SAME_TEMPLATE_CONTRAST
- DISTRACTOR_FIELD_MIX
- INTERMEDIATE_FIELD_CHAIN

The selected positive candidate must be:

```text
open_multi_field_final_all_markers
```

Candidate and control arms must reject single-field shortcuts:

- field A only
- field B only
- intermediate copy
- visible target bypass
- noisy distractor copy
- closed-pocket ablation
- wrong priority field
- prefix-only success

Positive requires:

- main final answer accuracy >= 0.55
- multi-field binding accuracy >= 0.55
- main pocket writeback rate >= 0.70
- main contrast group accuracy >= 0.55
- ablation final answer accuracy <= 0.20
- pocket ablation delta >= 0.30
- single-field shortcut rate = 0.0
- visible bypass violation rate = 0.0
- noisy distractor violation rate = 0.0
- direct `POCKET_VALUE=` marker rate = 0.0
- deterministic replay = true

If positive:

```text
decision = instnct_pocket_gated_multi_field_transfer_probe_positive
verdict = INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_POSITIVE
next = 141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM
```

Clean negatives:

- single_field_shortcut_detected -> 141B_SINGLE_FIELD_SHORTCUT_ANALYSIS
- multi_field_binding_failure -> 141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS
- pocket_ablation_not_decision_critical -> 141D_POCKET_CAUSALITY_FAILURE_ANALYSIS
- priority_conflict_failure -> 141E_PRIORITY_CONFLICT_FAILURE_ANALYSIS
- helper_integrity_failure -> 135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL

Boundaries:

- no training
- no source checkpoint mutation
- no helper/backend modification
- no public request-key change
- no runtime/release/product/deploy changes
- no root `LICENSE` change

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
