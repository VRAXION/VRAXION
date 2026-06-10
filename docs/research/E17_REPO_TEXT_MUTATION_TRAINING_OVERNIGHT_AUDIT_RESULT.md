# E17 Repo Text Mutation Training Overnight Audit Result

Status: completed.

## Decision

```text
decision = e17_repo_text_mutation_training_overnight_confirmed
next = E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM
primary_system = MUTATION_TRAINED_PRUNED_REPO_TEXT_POLICY_PRIMARY
positive_gate_passed = true
deterministic_replay_passed = true
checker_failure_count = 0
```

Run root:

```text
target/pilot_wave/e17_repo_text_mutation_training_overnight_audit/
```

## Budget

The requested overnight budget was accepted by the CLI, then downshifted by the
runner for the interactive Codex environment:

```text
requested_generations = 80
requested_population = 96
requested_train_episodes = 2000
requested_validation_episodes = 500
requested_heldout_episodes = 800

actual_generations = 18
actual_population = 48
actual_train_episodes = 900
actual_validation_episodes = 280
actual_heldout_episodes = 420

run_budget_class = partial_downshifted_interactive
downshift_reason = codex_interactive_runtime_downshift_from_overnight_budget
runtime_minutes = 0.954193
```

## Corpus And Split

```text
document_count = 298
chunk_count = 7151
train_file_count = 186
validation_file_count = 44
heldout_file_count = 68
train_episode_count = 900
validation_episode_count = 280
heldout_episode_count = 420
```

The split is by whole file. The checker verified no train/validation/heldout file
overlap.

## Training

```text
generations_completed = 18
candidate_count_evaluated = 864
checkpoint_count = 18
best_generation = 15
best_policy_id = pol_g14_ef07707631
pruned_policy_id = pruned_cc2a21e3ac
policy_complexity = 9
mutation_acceptance_rate = 0.586111
crossover_acceptance_rate = 0.413889
overfit_gap = -0.100595
pruned_cost_reduction = 0.000000
```

Required checkpoint artifacts were written, including `checkpoint_latest.json`,
`checkpoint_generation_<N>.json`, `best_policy_so_far.json`, and
`training_progress.jsonl`.

## Heldout Metrics

```text
exact_answer_accuracy = 1.000
canonical_object_accuracy = 0.900
evidence_chunk_accuracy = 1.000
retrieval_top1_accuracy = 1.000
field_extraction_accuracy = 1.000
metric_comparison_accuracy = 1.000
result_summary_accuracy = 1.000
cross_doc_chain_accuracy = 1.000
caveat_boundary_accuracy = 1.000
noisy_context_repair_accuracy = 1.000
long_context_memory_accuracy = 1.000
table_row_extraction_accuracy = 1.000
abstain_precision = 1.000
abstain_recall = 1.000
ambiguity_handling_accuracy = 1.000
hallucinated_answer_rate = 0.000
wrong_evidence_rate = 0.000
trace_validity = 1.000
renderer_faithfulness = 1.000
cost_per_episode = 5.421429
cost_per_chunk = 1.657205
```

Baseline deltas:

```text
delta_vs_bm25_exact_answer_accuracy = +0.428571
delta_vs_static_canonical_object_accuracy = +0.900000
```

## Failure Map

```text
best_task_family = FIELD_EXTRACTION
first_failing_family = none
likely_bottleneck = none
recommended_next_repair_milestone = E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM
```

## Source Audit

```text
source_fixture_audit_passed = true
aggregate_recomputed_from_episode_logs = true
deterministic_replay_passed = true
hand_authored_control_selected_as_primary = false
neural_dependencies_used = false
```

## Boundary

This is an overnight real-repository-text mutation-training audit for a controlled Flow text policy. It uses real local project documents, but task wrappers and labels are deterministically generated from those documents. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Verification

```text
python3 -m py_compile scripts/probes/run_e17_repo_text_mutation_training_overnight_audit.py scripts/probes/run_e17_repo_text_mutation_training_overnight_audit_check.py
python3 scripts/probes/run_e17_repo_text_mutation_training_overnight_audit.py --out target/pilot_wave/e17_repo_text_mutation_training_overnight_audit --generations 80 --population 96 --train-episodes 2000 --validation-episodes 500 --heldout-episodes 800 --checkpoint-every 1 --max-runtime-minutes 360 --resume
python3 scripts/probes/run_e17_repo_text_mutation_training_overnight_audit_check.py --out target/pilot_wave/e17_repo_text_mutation_training_overnight_audit --write-summary
```
