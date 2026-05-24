# STABLE_LOOP_PHASE_LOCK_143F_INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRM Result

143F implements the scale confirm and resolved-marker dependency test after the positive 143A smoke. The runner is executable helper-only: it may call `shared_raw_generation_helper.raw_generate` through the shared helper, while the checker must not call generation.

This confirms constrained helper/backend multi-pocket arbitration scale stability and resolved-final-marker dependency only. It is not open-ended reasoning. It is not general composition. It is not GPT-like/open-domain/broad assistant capability. It is not production/public API/deployment/safety readiness. It is not architecture superiority.

## What 143F Must Decide

143F has two acceptable pass states.

If resolved-marker-present scale and the no-resolved bridge both pass:

```text
decision = instnct_pocket_gated_multi_pocket_arbitration_scale_confirmed
verdict = INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_SCALE_CONFIRMED
next = 144A_INSTNCT_POCKET_GATED_RULE_BOUND_ARBITRATION_TRANSFER_PROBE
```

If resolved-marker-present scale passes but the no-resolved subset fallback-fails cleanly:

```text
decision = resolved_final_marker_dependency_confirmed
verdict = INSTNCT_POCKET_GATED_MULTI_POCKET_ARBITRATION_DEPENDS_ON_RESOLVED_FINAL_MARKER
next = 143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS
```

The expected likely outcome is the clean dependency route, because the current helper scans configured payload markers and returns the first value after a present marker. If `FINAL_MARKERS` are kept static but no final marker appears in the prompt, the helper is expected to use the closed-pocket fallback.

## Required Evidence

The smoke run must write:

```text
resolved_marker_present_subset_report.json
no_resolved_final_marker_subset_manifest.json
no_resolved_final_marker_subset_results.jsonl
no_resolved_final_marker_subset_scoring.jsonl
no_resolved_final_marker_subset_report.json
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
decision.json
summary.json
report.md
```

The no-resolved subset is valid only if the checker proves:

```text
no_resolved_unique_checkpoint_path_count = 1
no_resolved_unique_checkpoint_hash_count = 1
no_resolved_manifest_payload_markers_static = true
no_resolved_per_row_manifest_switch_rate = 0.0
no_resolved_per_row_payload_marker_switch_rate = 0.0
```

The checker also rejects hidden renamed final/winner/gold/answer markers in no-resolved prompts.

## Interpretation Boundary

If `NO_RESOLVED_FINAL_MARKER_SUBSET` fails cleanly, the result is not an architecture failure. It means the current constrained helper/backend path depends on resolved final payload markers. The next step is an artifact-only failure analysis of that helper-semantics bottleneck.

All broad capability/readiness flags remain false.
