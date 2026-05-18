# STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM Result

079 implemented the eval-only fresh confirmation gate for the 078
`TOKEN_COMPOSITION_REPAIR` checkpoint.

The smoke run completed and wrote all required artifacts under:

```text
target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke
```

## Verdict

The 079 smoke result is an honest fail:

```text
CHAT_COMPOSITION_FRESH_CONFIRM_FAILS
TEMPLATE_COPY_DETECTED
```

Integrity checks passed:

```text
upstream_078_summary_present = true
upstream_078_positive = true
checkpoint_exists = true
eval_started_after_079_start = true
checkpoint_hash_unchanged = true
train_step_count = 0
prediction_oracle_used = false
llm_judge_used = false
decoder_path = token_level_next_token
response_table_used_for_main_prediction = false
```

Fresh prompt leakage was rejected:

```text
overlap_with_078_train_prompt_count = 0
overlap_with_078_eval_prompt_count = 0
overlap_with_077_prompt_count = 0
overlap_with_076_prompt_count = 0
near_duplicate_prompt_count = 0
max_prompt_token_jaccard_vs_078_train = 0.6111111111111112
max_prompt_token_jaccard_vs_078_eval = 0.75
max_prompt_token_jaccard_vs_077 = 0.625
max_prompt_token_jaccard_vs_076 = 0.7333333333333333
```

The bounded chat behavior still produced non-empty multi-token outputs and kept
context slots plus finite-label retention intact:

```text
multi_token_response_rate = 1.0
non_empty_response_rate = 1.0
fresh_instruction_accuracy = 1.0
fresh_context_carry_accuracy = 1.0
slot_binding_accuracy = 1.0
two_turn_dialogue_accuracy = 1.0
boundary_refusal_accuracy = 1.0
finite_label_retention_accuracy = 1.0
empty_output_rate = 0.0
space_output_rate = 0.0
static_response_rate = 0.0
repetition_rate = 0.0
```

The failure is specifically template-copy regression under the stricter 079
audit:

```text
novel_response_rate = 0.0
template_copy_rate = 1.0
exact_train_response_copy_rate = 1.0
exact_eval_response_copy_rate = 0.7333333333333333
response_table_copy_rate = 0.13333333333333333
semantic_template_overlap_rate = 0.8666666666666667
```

## Interpretation

079 shows that the 078 repair improved over the earlier 076/077 response-table
failure, but the stricter fresh-confirm audit still sees outputs as copied from
the 078 repair response surface. This is not a prompt leakage issue, not a
checkpoint mutation issue, not a training side effect, and not a collapse issue.

079 does not prove GPT-like assistant readiness. It proves no full English LM
behavior, no language grounding, no production chat, no safety alignment,
no public beta, no GA, and no hosted SaaS readiness.

## Next

Because 079 failed on template-copy detection, the next milestone is:

```text
079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS
```

That analysis should attribute the 079 copy signal across exact 078 train
responses, 078 eval/generated outputs, semantic template overlap, and the
remaining response-table surface before another repair attempt.
