# E16C Text-Flow Micro-Training Ladder Failure Map Result

Status: completed.

## Decision

```text
decision = e16c_text_flow_micro_training_ladder_partial_confirmed
next = E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR
primary_system = MICRO_TRAINING_PRUNED_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e16c_text_flow_micro_training_ladder_failure_map/
```

## Failure Map

```text
best_stage_passed = 6
first_failing_stage = 7
first_failing_stage_name = MULTI_SENTENCE_BINDING_MEMORY
failure_signature = stage_7_memory_binding_capacity_shortfall
failure_reason_code = finite_memory_slots_and_delayed_binding_policy_insufficient
recommended_next_repair = E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR
```

## Stage Table

| stage | name | pass | key metrics |
|---:|---|---:|---|
| 0 | RAW_CHAR_STREAM_RECOVERY | true | char_stream_recovery_accuracy=1.000 |
| 1 | TOKEN_BOUNDARY_DISCOVERY | true | token_boundary_accuracy=0.986, token_recovery_accuracy=0.986 |
| 2 | TOKEN_COPY_AND_ORDER | true | order_program_discovery_accuracy=0.912, output_sequence_accuracy=0.944 |
| 3 | WORD_LEVEL_REWRITE_EVIDENCE | true | rewrite_evidence_fit_accuracy=0.902, heldout_rewrite_accuracy=0.875 |
| 4 | FILTER_AND_DECOY_HANDLING | true | filter_program_accuracy=0.895, decoy_rejection_rate=0.906, wrong_writeback_rate=0.020 |
| 5 | PHRASE_COMPOSITION | true | phrase_composition_accuracy=0.833, chain_order_accuracy=0.822 |
| 6 | CONTROLLED_SENTENCE_TEMPLATE | true | sentence_template_accuracy=0.778, heldout_template_accuracy=0.722 |
| 7 | MULTI_SENTENCE_BINDING_MEMORY | false | multi_sentence_binding_accuracy=0.622, long_horizon_recall=0.611, ambiguous_abstain_accuracy=0.778 |
| 8 | NOISY_MULTI_SENTENCE_REPAIR | false | repair_success_rate=0.556, noise_rejection_rate=0.681, canonical_decoder_exact_accuracy=0.708, trace_validity=0.883 |

## Primary Metrics

```text
discovered_library_size = 8
discovered_program_count = 216
average_program_len = 5.750
max_program_len = 7
heldout_vocab_accuracy = 0.874
randomized_codebook_generalization = 0.906
heldout_template_accuracy = 0.722
heldout_composition_accuracy = 0.812
trace_validity = 0.964
wrong_writeback_rate = 0.021
destructive_overwrite_rate = 0.007
branch_contamination_rate = 0.000
semantic_slot_leak_detected = false
macro_leak_detected = false
privileged_control_selected_as_primary = false
```

## Ablations

```text
no_rewrite_ablation_first_failed_stage = 3
no_validity_ablation_first_failed_stage = 4
no_memory_ablation_first_failed_stage = 7
no_conditional_ablation_first_failed_stage = 6
too_short_program_ablation_first_failed_stage = 5
```

The ablations are coherent with the ladder: removing rewrite evidence blocks the
rewrite stage, removing validity evidence blocks decoy handling, removing memory
keeps early stages intact but fails at multi-sentence binding, removing
conditionals fails at controlled templates, and a too-short program budget fails
composition.

## Interpretation

E16C is a partial informative pass. The micro-program ladder reaches controlled
sentence templates from the minimal micro-VM, but does not clear the
multi-sentence binding memory gate. The next repair should focus on finite
memory slot capacity, delayed binding trace policy, and ambiguity repair search
around Stage 7.

## Boundary

This is a deterministic synthetic controlled text-flow micro-training ladder. It maps how far micro-program discovery gets from a minimal micro-VM. It does not prove general natural language AI or unconstrained invention from absolute nothing.

## Verification

```text
python3 -m py_compile scripts/probes/run_e16c_text_flow_micro_training_ladder_failure_map.py scripts/probes/run_e16c_text_flow_micro_training_ladder_failure_map_check.py
python3 scripts/probes/run_e16c_text_flow_micro_training_ladder_failure_map.py --out target/pilot_wave/e16c_text_flow_micro_training_ladder_failure_map
python3 scripts/probes/run_e16c_text_flow_micro_training_ladder_failure_map_check.py --out target/pilot_wave/e16c_text_flow_micro_training_ladder_failure_map --write-summary
```
