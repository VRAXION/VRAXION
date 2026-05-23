# STABLE_LOOP_PHASE_LOCK_142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE

142A is the executable helper-only conflict/priority transfer probe after the
141Z decision plan.

It must use `shared_raw_generation_helper.py` for final generation, with helper
requests containing only:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

142A does not train, mutate source checkpoints, modify
`shared_raw_generation_helper.py`, change helper/backend request keys, deploy,
alter runtime/release/product surfaces, or change root `LICENSE`.

Required upstream:

```text
141Z decision = conflict_priority_transfer_probe_recommended
141Z next = 142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE
141F decision = instnct_pocket_gated_multi_field_transfer_scale_confirmed
```

The probe must include A wins rows, B wins rows, table override wins rows, rule
override wins rows, visible value loses rows, noisy distractor loses rows,
same-template different-priority contrast rows, and priority inversion pairs.
It must not be an always B wins task, an always A wins task, or a table/rule
default shortcut task.

Required hardening artifacts include:

- `priority_rule_manifest.json`
- `conflict_pair_manifest.json`
- `priority_inversion_report.json`
- `wrong_priority_field_report.json`
- `priority_control_report.json`
- `helper_request_audit.json`
- `canonical_metric_alias_report.json`
- `per_seed_gate_report.json`
- `per_family_gate_report.json`

Positive route:

```text
decision = instnct_pocket_gated_conflict_priority_transfer_probe_positive
verdict = INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_POSITIVE
next = 142F_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_SCALE_CONFIRM
```

Boundary: constrained helper/backend conflict-priority final selection only,
not open-ended reasoning, not general composition, not GPT-like readiness, not
open-domain reasoning, not broad assistant capability, not production/public
API/deployment/safety readiness, and not architecture superiority.

