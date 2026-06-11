# E18B_FULL_BUDGET_REPO_TEXT_STRESS_CONFIRM Result

## Online Codex status

This result document records the online Codex smoke/preflight execution for the E18B runner/checker. The online execution is intentionally bounded and must not be interpreted as a full-budget confirmation unless the full minimum budget actually ran.

## Boundary

This is a real-repository-text stress and latency audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Expected online decision

The online run is expected to produce one of the non-full decisions, normally:

```text
decision=e18b_full_budget_repo_text_stress_preflight_confirmed
```

If the smoke run or checker fails, acceptable non-full decisions are partial, partial-downshifted, failed, or invalid/incomplete. A full confirmed decision is forbidden unless all full-confirm minimums and gates are actually satisfied.

## Artifact location

The runner writes E18B artifacts under:

```text
target/pilot_wave/e18b_full_budget_repo_text_stress_confirm/
```

The authoritative machine-readable result is `summary.json`, with `decision.json`, `aggregate_metrics.json`, per-episode logs, corpus/split reports, policy reports, ablation reports, checkpoint reports, latency reports, and checker summaries used for audit and recomputation.

## Local full-budget follow-up

Run the strict/no-downshift command from the contract on a local machine or Deck/PC overnight. The checker must be run after the runner and must report zero checker failures before any full confirmation may be considered.

## Online Codex smoke/preflight snapshot

The bounded online run completed as a preflight only:

- `decision = e18b_full_budget_repo_text_stress_preflight_confirmed`
- `next = RUN_E18B_FULL_BUDGET_LOCALLY_WITH_STRICT_NO_DOWNSHIFT`
- `primary_system = MUTATION_TRAINED_PRUNED_STRESS_POLICY_PRIMARY`
- `positive_gate_passed = true`
- `checker_failure_count = 0`
- `run_budget_class = smoke_preflight`
- `full_confirmation_forbidden = true`
- `source_fixture_audit_passed = true`
- `aggregate_recomputed_from_episode_logs = true`
- `generations_completed = 3`
- `candidate_count_evaluated = 24`
- `checkpoint_count = 3`
- `heldout_episode_count = 80`
- `stress_episode_count = 80`
- `exact_answer_accuracy = 0.962500`
- `no_source_path_accuracy = 1.000000`
- `paraphrased_field_accuracy = 1.000000`
- `same_key_conflict_accuracy = 1.000000`
- `same_milestone_distractor_accuracy = 1.000000`
- `target_not_first_accuracy = 1.000000`
- `noisy_context_repair_accuracy = 0.833333`
- `long_context_memory_accuracy = 1.000000`
- `latency_p50_ms = 0.214589`
- `latency_p95_ms = 0.439993`
- `latency_max_ms = 0.477071`

The full-confirm minimums were not met in this online run, so a full E18B confirmation remains forbidden.
