# E16B Text-Flow Operator Composition Discovery Confirm Result

Status: completed.

## Decision

```text
decision = e16b_text_flow_operator_composition_discovery_confirmed
next = E16C_TEXT_FLOW_OPERATOR_INVENTION_FROM_MICRO_PRIMITIVES_CONFIRM
primary_system = COMPOSITION_DISCOVERY_PRUNED_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e16b_text_flow_operator_composition_discovery_confirm/
```

## What Was Tested

E16B removes direct macro operators from the primary discovery grammar and tests
whether bounded primitive chains can be searched, selected, and pruned from
support/query evidence. The runtime uses anonymized token streams, explicit
mapping-table evidence for `MAP`, invalid/decoy evidence for `FILTER_VALID`, and
gated writeback for ambiguous or unsafe cases.

## Primary Metrics

```text
discovered_library_size = 8
discovered_program_count = 143
average_program_chain_len = 5.000
max_program_chain_len = 7
discovery_test_exact_accuracy = 1.000
composition_exact_accuracy = 1.000
heldout_vocab_accuracy = 1.000
randomized_codebook_generalization = 1.000
support_fit_accuracy = 1.000
program_selection_accuracy = 1.000
support_disambiguation_accuracy = 1.000
ambiguous_case_abstain_or_repair_accuracy = 1.000
trace_validity = 1.000
wrong_writeback_rate = 0.000
destructive_overwrite_rate = 0.000
branch_contamination_rate = 0.000
macro_removed_confirmed = true
direct_macro_leak_detected = false
semantic_slot_leak_detected = false
privileged_control_selected_as_primary = false
cost_per_tick = 2.120
pruned_cost_reduction_vs_unpruned = 0.489
```

## System Comparison

| system | exact | comp | trace | wrong | cost/tick |
|---|---:|---:|---:|---:|---:|
| RANDOM_LIBRARY_SMALL | 0.091 | 0.000 | 0.091 | 0.000 | 0.642 |
| RANDOM_LIBRARY_MATCHED_BUDGET | 0.091 | 0.000 | 0.091 | 0.000 | 0.429 |
| RANDOM_LIBRARY_BEST_OF_N_CONTROL | 0.091 | 0.000 | 0.091 | 0.000 | 0.702 |
| TRUE_MACRO_LIBRARY_CONTROL | 1.000 | 1.000 | 1.000 | 0.000 | 7.500 |
| TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL | 1.000 | 1.000 | 1.000 | 0.000 | 2.120 |
| COMPOSITION_DISCOVERY_NO_GATE | 0.727 | 0.800 | 0.727 | 0.273 | 2.093 |
| COMPOSITION_DISCOVERY_PRIMARY | 1.000 | 1.000 | 1.000 | 0.000 | 4.145 |
| COMPOSITION_DISCOVERY_PRUNED_PRIMARY | 1.000 | 1.000 | 1.000 | 0.000 | 2.120 |
| INSUFFICIENT_CHAIN_LEN_ABLATION | 0.364 | 0.300 | 0.364 | 0.000 | 1.248 |
| MISSING_ORDER_PRIMITIVES_ABLATION | 0.091 | 0.000 | 0.091 | 0.000 | 0.036 |
| MISSING_MAP_PRIMITIVE_ABLATION | 0.364 | 0.300 | 0.364 | 0.000 | 2.030 |
| MISSING_FILTER_PRIMITIVE_ABLATION | 0.818 | 0.800 | 0.818 | 0.000 | 1.917 |
| AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION | 0.909 | 1.000 | 0.909 | 0.091 | 2.202 |

## Ablations

```text
insufficient_chain_len_exact_accuracy = 0.364
missing_order_primitives_exact_accuracy = 0.091
missing_map_primitive_exact_accuracy = 0.364
missing_filter_primitive_exact_accuracy = 0.818
ambiguous_no_abstain_wrong_commit_rate = 1.000
```

The ablations satisfy the expected dependency checks: short chains cannot
express the composed programs, missing order primitives fail reverse/order
families, missing `MAP` fails map families, missing `FILTER_VALID` fails
filter/decoy families, and removing ambiguity abstain causes wrong commits.

## Interpretation

E16B confirms that, in this deterministic controlled proxy, the primary can
discover a macro-free primitive-chain library and prune it from 143 candidate
programs to 8 selected programs while preserving heldout exactness, trace
validity, and gated writeback safety. The privileged macro control remains
invalid as primary, and the hand-authored primitive control is a reference arm,
not the selected discovery arm.

## Boundary

This confirms grammar-level operator composition discovery from lower-level primitives in a deterministic synthetic controlled text-flow proxy. It does not confirm unconstrained operator invention or general natural-language AI.

## Verification

```text
python3 -m py_compile scripts/probes/run_e16b_text_flow_operator_composition_discovery_confirm.py scripts/probes/run_e16b_text_flow_operator_composition_discovery_confirm_check.py
python3 scripts/probes/run_e16b_text_flow_operator_composition_discovery_confirm.py --out target/pilot_wave/e16b_text_flow_operator_composition_discovery_confirm
python3 scripts/probes/run_e16b_text_flow_operator_composition_discovery_confirm_check.py --out target/pilot_wave/e16b_text_flow_operator_composition_discovery_confirm --write-summary
```
