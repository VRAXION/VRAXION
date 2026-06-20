# E96 Trace/Ground Memory Hygiene Expansion Contract

## Purpose

Expand the Operator Library with Trace Ledger / Ground Field hygiene skills:
deduplication, provenance preservation, delayed feedback integration,
contradiction indexing, scoped Ground-promotion candidates, stale pruning,
scope lifetime handling, and replay-hash audit.

Boundary:

```text
controlled Trace/Ground memory hygiene proxy
not persistent user memory
not open-domain memory system
```

## Required Operator Candidates

```text
trace_deduplication_lens
provenance_chain_guard
delayed_feedback_integrator_t_stab
contradiction_memory_index_lens
ground_promotion_candidate_scribe
stale_trace_pruner_guard
scope_lifetime_t_stab
replay_hash_audit_guard
```

## Required Controls

```text
duplicate_trace_accumulator
provenance_dropping_committer
delayed_feedback_ignorer
contradiction_forgetting_committer
always_promote_to_ground
stale_trace_keeper
always_prune_control
trace_dedup_clone
```

## Positive Decision

```text
decision = e96_trace_ground_memory_hygiene_expansion_confirmed
```

Requires:

```text
validation_memory_hygiene_success_min = 1.0
adversarial_memory_hygiene_success_min = 1.0
validation_provenance_validity_min = 1.0
validation_replay_safe_min = 1.0
adversarial_bad_ground_promotion_max = 0.0
adversarial_stale_pollution_max = 0.0
adversarial_false_prune_max = 0.0
checker failure_count = 0
sample-only checker failure_count = 0
```

## Commands

```text
python private_probe_runner_removed
python private_probe_runner_removed --out target/pilot_wave/e96_trace_ground_memory_hygiene_expansion --write-summary
python private_probe_runner_removed --sample-only archived_public_artifact_sample_removed --write-summary
```
