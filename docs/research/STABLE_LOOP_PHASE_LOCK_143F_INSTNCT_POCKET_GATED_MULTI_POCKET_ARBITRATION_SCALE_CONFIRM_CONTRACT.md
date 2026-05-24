# STABLE_LOOP_PHASE_LOCK_143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM Contract

143F is an executable helper-only scale confirm after positive 143A. It is not just a larger 143A run: it must answer whether the current constrained helper/backend path still works when resolved final payload markers are removed.

This confirms constrained helper/backend multi-pocket arbitration scale stability and resolved-final-marker dependency only. It is not open-ended reasoning. It is not general composition. It is not GPT-like/open-domain/broad assistant capability. It is not production/public API/deployment/safety readiness. It is not architecture superiority.

## Boundaries

- No training.
- No source checkpoint mutation.
- No helper/backend/request-key changes.
- No runtime/release/product/deploy/public API changes.
- No root `LICENSE` change.
- Runner may call `shared_raw_generation_helper.raw_generate` only through the shared helper.
- Checker must not call `raw_generate`.
- Helper requests may include only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config`.
- Expected/scorer/oracle metadata must not enter helper requests.
- All broad capability/readiness flags remain false.

## Required Upstream

Require 143A:

```text
decision = instnct_pocket_gated_multi_pocket_arbitration_probe_positive
next = 143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM
main_final_answer_accuracy = 1.0
multi_pocket_arbitration_accuracy = 1.0
main_pocket_writeback_rate = 1.0
ablation_final_answer_accuracy = 0.0
pocket_ablation_delta_final_answer_accuracy = 1.0
resolved_final_marker_echo_rate = 0.0
resolved_final_marker_echo_control_failed = true
shortcut/bypass/direct marker rates = 0.0
deterministic_replay_passed = true
```

## Subsets

`RESOLVED_FINAL_MARKER_PRESENT_SUBSET` is the scaled 143A main path. It uses a static selected candidate manifest with `FINAL_MARKERS`, and the resolved final payload marker contains the expected selected pocket value.

`NO_RESOLVED_FINAL_MARKER_SUBSET` removes the final payload marker line entirely while keeping the same static selected candidate manifest with `FINAL_MARKERS`. It must not use per-row checkpoint manifests, per-row payload marker lists, hidden selected markers, or any renamed final/winner/gold/answer marker.

`NO_RESOLVED_EXPLICIT_WINNER_LABEL_SUBSET` may contain `winner=pocket_b` style text.

`NO_RESOLVED_RULE_DERIVED_WINNER_SUBSET` must not contain explicit `selected_pocket_id` or `winner=pocket_*` text.

`NO_RESOLVED_ABC_STATIC_MARKER_CONTROL` is diagnostic only. It removes final markers and uses the same static marker list for all rows:

```text
pocket A candidate:
pocket B candidate:
pocket C candidate:
```

## No-Resolved Scanner

The checker must regex-scan no-resolved prompts case-insensitively and reject these marker equivalents:

```text
arbitrated[-_ ]?final
selected[-_ ]?final
winner[-_ ]?value
final[-_ ]?target
expected[-_ ]?answer
gold[-_ ]?output
answer[-_ ]?value
resolved[-_ ]?final
ANSWER\s*=
TARGET\s*=
GOLD\s*=
```

## Pass States

The checker accepts only two successful states.

Full positive:

```text
decision = instnct_pocket_gated_multi_pocket_arbitration_scale_confirmed
verdict = INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRMED
next = 144A_INSTNCT_POCKET_GATED_RULE_BOUND_ARBITRATION_TRANSFER_PROBE
```

Clean dependency:

```text
decision = resolved_final_marker_dependency_confirmed
verdict = INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_DEPENDS_ON_RESOLVED_FINAL_MARKER
next = 143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS
```

Clean dependency is valid only when resolved-marker-present scale passes, no-resolved rows fail through fallback, shortcut/visible/noisy/train-namespace rates remain zero, static-manifest checks pass, and the dependency delta is at least `0.25`.

## Required Artifacts

143F must write the standard helper/eval artifacts plus:

```text
no_resolved_final_marker_subset_manifest.json
no_resolved_final_marker_subset_results.jsonl
no_resolved_final_marker_subset_scoring.jsonl
no_resolved_final_marker_subset_report.json
resolved_marker_present_subset_report.json
resolved_marker_dependency_report.json
no_resolved_final_marker_shortcut_report.json
no_resolved_abc_static_marker_control_report.json
no_resolved_explicit_winner_label_subset_report.json
no_resolved_rule_derived_winner_subset_report.json
helper_request_audit.json
canonical_metric_alias_report.json
per_seed_gate_report.json
per_family_gate_report.json
per_pocket_gate_report.json
```

If no-resolved fails cleanly, it must be reported as a helper-semantics bottleneck, not an architecture failure.
