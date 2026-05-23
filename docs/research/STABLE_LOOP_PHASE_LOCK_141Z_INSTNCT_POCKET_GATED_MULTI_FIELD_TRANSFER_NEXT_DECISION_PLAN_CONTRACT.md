# STABLE_LOOP_PHASE_LOCK_141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN

141Z is the planning-only decision milestone after hardened positive 141F.

It verifies the hardened 141F evidence chain and writes the machine-readable
plan for the next executable probe:

```text
142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE
```

141Z must not train, call helper generation, call `raw_generate`, mutate
checkpoints, modify `shared_raw_generation_helper.py`, change helper/backend or
public request keys, deploy, alter runtime/release/product surfaces, or change
root `LICENSE`.

Expected decision:

```text
decision = conflict_priority_transfer_probe_recommended
selected_option = conflict_priority_transfer
next = 142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE
```

Required upstream 141F hardening:

- helper_request_audit.json
- canonical_metric_alias_report.json
- per_seed_gate_report.json
- per_family_gate_report.json

141Z must require allowed helper request keys only, zero forbidden helper
metadata, runner-side helper generation allowed only in 141F, checker-side
helper generation forbidden, and passing per-seed and per-family gate reports.

The target 142A plan must include A-wins, B-wins, table override, rule override,
visible-loses, noisy-distractor-loses, same-template different-priority contrast
rows, and priority inversion pairs. It must explicitly reject always-B,
always-A, priority-default, single-field, visible/noisy, and closed-pocket
shortcuts.

Boundary: constrained helper/backend evidence only, not open-ended reasoning,
not general composition, not GPT-like readiness, not open-domain reasoning, not
broad assistant capability, not production/public API/deployment/safety
readiness, and not architecture superiority.

Boundary phrase: not broad assistant capability.
