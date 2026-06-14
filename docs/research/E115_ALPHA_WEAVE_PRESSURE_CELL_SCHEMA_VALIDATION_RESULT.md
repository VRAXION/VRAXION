# E115 Alpha-Weave Pressure Cell Schema Validation Result

```text
decision = e115_alpha_weave_pressure_cell_schema_confirmed
checker_failure_count = 0
schema_version = AlphaWeavePressureCell-v1
```

## What Was Locked

E115 locks the first alpha-Weave pressure-cell format:

```text
public_input
  visible to candidate/runtime

hidden_oracle
  checker/trainer only

training_metadata
  target_skill / target_operators / budget
  hidden from candidate

adversarial_variants
  counterfactual pack for shortcut attacks
```

The candidate-visible path is `public_input` only. Target skill names,
target operator names, expected answers, oracle traces, and forbidden behavior
must not appear in `public_input`.

## Main Metrics

```text
schema_validity = true
oracle_leak_rate = 0.000000
target_operator_leak_rate = 0.000000
variant_count = 12
primary_success_rate = 1.000000
action_accuracy = 1.000000
answer_accuracy = 1.000000
citation_exact_rate = 1.000000
trace_dependency_coverage = 1.000000
false_commit_rate = 0.000000
wrong_scope_call_rate = 0.000000
unsupported_answer_rate = 0.000000
over_budget_rate = 0.000000
controls_all_invalid_as_general_policy = true
```

## Sample Scenario

The locked sample pack is a DnD-like dungeon evidence case:

```text
The party tracks whether the moon-rune lever is safe right now.
Evidence can be old, weak, replayed, conflicting, or inactive narrative text.
```

The pack includes:

```text
answerable_base
missing_evidence
weak_source
unresolved_conflict
stale_replay
source_trust_inversion
order_swap
quote_or_inactive_scope
negative_scope_story_text
citation_id_shortcut_trap
citation_span_trap
overbudget_fullscan_trap
```

## Controls

The following controls were invalidated by the adversarial pack:

```text
label_leak_control
latest_only_control
source_name_shortcut_control
citation_id_shortcut_control
full_scan_control
answer_without_trace_control
negative_scope_overcall_control
```

## Interpretation

The alpha-Weave v1 schema is ready as a stable pressure-cell unit for the next
targeted data generation step. E115 does not train the runtime on these cells;
it only proves that the cell format can carry targeted pressure without leaking
the target/operator/oracle information into the visible input.

Boundary: schema/curriculum-unit validation only. No final-training,
PermaCore, TrueGolden, or open-domain reasoning claim.
