# E18 Repo Text Policy Stress And Latency Confirm Result

Status: completed.

## Decision

```text
decision = e18_repo_text_policy_stress_and_latency_partial_downshifted
next = E18B_FULL_BUDGET_REPO_TEXT_STRESS_CONFIRM
primary_system = MUTATION_TRAINED_PRUNED_STRESS_POLICY_PRIMARY
positive_gate_passed = false
deterministic_replay_passed = true
checker_failure_count = 0
run_budget_class = partial_downshifted
```

Run root:

```text
target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm/
```

## Budget

The requested stress budget was accepted by the CLI, then downshifted for the
interactive Codex run. Because the actual budget is below the full-confirm
minimums, the result is intentionally not a full confirmation.

```text
requested_generations = 80
requested_population = 96
requested_train_episodes = 2500
requested_validation_episodes = 700
requested_heldout_episodes = 1000
requested_stress_episodes = 1000

actual_generations = 3
actual_population = 8
actual_train_episodes = 120
actual_validation_episodes = 50
actual_heldout_episodes = 80
actual_stress_episodes = 80

downshift_reason = codex_interactive_runtime_downshift_from_e18_stress_budget
runtime_minutes = 0.163928
```

## Corpus And Split

```text
document_count = 298
chunk_count = 7163
train_file_count = 188
validation_file_count = 39
heldout_file_count = 53
stress_file_count = 18
train_episode_count = 120
validation_episode_count = 50
heldout_episode_count = 80
stress_episode_count = 80
```

The split is by whole file with a separate adversarial stress split. The checker
verified no train/validation/heldout/stress file overlap.

## Training

```text
generations_completed = 3
candidate_count_evaluated = 24
checkpoint_count = 3
best_generation = 2
best_policy_id = pol_13db144949
pruned_policy_id = pruned_cc29a30a60
policy_complexity = 17
mutation_acceptance_rate = 0.583333
crossover_acceptance_rate = 0.416667
overfit_gap = -0.482176
pruned_cost_reduction = 0.819479
```

## Heldout And Stress Metrics

```text
exact_answer_accuracy = 0.687500
canonical_object_accuracy = 0.687500
evidence_chunk_accuracy = 1.000000
retrieval_top1_accuracy = 1.000000
no_source_path_accuracy = 0.666667
paraphrased_field_accuracy = 0.583333
same_key_conflict_accuracy = 0.416667
same_milestone_distractor_accuracy = 0.416667
target_not_first_accuracy = 0.472222
table_row_extraction_accuracy = 1.000000
metric_delta_accuracy = 1.000000
noisy_context_repair_accuracy = 0.400000
long_context_memory_accuracy = 0.416667
ambiguity_handling_accuracy = 1.000000
hallucinated_answer_rate = 0.000000
wrong_evidence_rate = 0.000000
trace_validity = 0.953125
renderer_faithfulness = 1.000000
```

Latency:

```text
latency_p50_ms = 0.036020
latency_p95_ms = 0.077311
latency_max_ms = 0.116520
episodes_per_second = 24625.117219
```

Baseline and hint deltas:

```text
delta_vs_bm25_no_source_path_accuracy = +0.473334
delta_vs_static_same_key_conflict_accuracy = +0.416667
source_path_hint_dependency_delta = +0.333333
field_name_hint_dependency_delta = +0.333333
```

## Failure Map

```text
first_failing_family = NO_SOURCE_PATH_FIELD_EXTRACTION
likely_bottleneck = budget downshift + source_path reliance
recommended_next_repair = E18B_FULL_BUDGET_REPO_TEXT_STRESS_CONFIRM
```

The stress run shows that table/numeric extraction, ambiguity handling, evidence
tracking, and latency are coherent in this partial budget run. The main weakness
is the less-hinted retrieval/extraction regime: no-source-path, paraphrase,
same-key conflict, same-milestone distractor, target-not-first, noisy context, and
long-context families remain below the full E18 gate.

## Source Audit

```text
source_fixture_audit_passed = true
aggregate_recomputed_from_episode_logs = true
deterministic_replay_passed = true
source_path_oracle_selected_as_primary = false
field_name_oracle_selected_as_primary = false
hand_authored_extractor_selected_as_primary = false
neural_dependencies_used = false
```

## Boundary

This is a real-repository-text stress and latency audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Verification

```text
python3 -m py_compile scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm.py scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm_check.py
python3 scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm.py --out target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm --generations 80 --population 96 --train-episodes 2500 --validation-episodes 700 --heldout-episodes 1000 --stress-episodes 1000 --checkpoint-every 1 --max-runtime-minutes 360 --resume
python3 scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm_check.py --out target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm --write-summary
```
