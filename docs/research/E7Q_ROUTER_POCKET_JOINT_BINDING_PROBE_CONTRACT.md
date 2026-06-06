# E7Q Router-Pocket Joint Binding Probe Contract

## Purpose

`E7Q_ROUTER_POCKET_JOINT_BINDING_PROBE` follows E7P.

E7P froze the router and teacher-forced the expected route while training the numeric pocket interface. E7Q tests the next sharper question:

```text
Can the control/router layer and a numeric pocket library learn together,
while the result remains a reusable callable pocket system instead of a
private router-pocket shortcut protocol?
```

## Systems

```text
frozen_router_trained_pocket
trained_router_frozen_pocket
trained_router_trained_pocket
trained_router_trained_pocket_slot_guard
full_end_to_end_training_control
random_router_control
oracle_route_reference
```

`full_end_to_end_training_control` is diagnostic only. It is not the primary architecture.

## Training Scope

```text
frozen_router_trained_pocket:
  E7P-style slot-contract pocket training
  expected route remains frozen/teacher-forced

trained_router_frozen_pocket:
  train typed route head only
  pocket library remains frozen

trained_router_trained_pocket:
  train route head + numeric pocket library jointly
  no strong slot guard

trained_router_trained_pocket_slot_guard:
  train route head + numeric pocket library jointly
  stronger route, preservation, and result-slot losses

full_end_to_end_training_control:
  monolithic diagnostic control

random_router_control:
  deterministic random route mapping

oracle_route_reference:
  symbolic oracle route/reference only
```

## Reuse Gate

Joint binding is not enough by itself. A joint system must also show that the pocket remains callable after binding:

```text
bound_usefulness:
  trained router + trained pockets

pocket_reuse_after_binding_usefulness:
  oracle route + trained pockets

router_transfer_to_baseline_usefulness:
  trained router + baseline pockets

private_protocol_risk:
  bound_usefulness - min(reuse, router_transfer)
```

If joint binding improves only when the router and pocket stay together, but reuse collapses, the result is treated as private protocol risk rather than reusable pocket learning.

## Metrics

```text
composition usefulness
answer accuracy
route accuracy
heldout/OOD/counterfactual/adversarial usefulness
state preservation error
result-slot corruption rate
next-pocket input compatibility error
reusability after binding
private protocol risk
router transfer to baseline
parameter count
bit budget
deterministic replay hash match
checker failure_count
```

## Required Artifacts

```text
backend_manifest.json
task_generation_report.json
baseline_pocket_training_report.json
router_training_report.json
joint_binding_training_report.json
flow_contract_report.json
reuse_after_binding_report.json
private_protocol_leakage_report.json
composition_report.json
error_attribution_report.json
system_results.json
leakage_report.json
deterministic_replay.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
```

Long runs must write progress and heartbeat artifacts continuously. No run may depend only on final writeout.

## Decision Labels

```text
e7q_router_pocket_joint_binding_positive
e7q_slot_guard_joint_binding_positive
e7q_slot_guard_improves_but_not_solved
e7q_router_discovery_not_interface_fix
e7q_private_router_pocket_protocol_detected
e7q_full_end_to_end_control_preferred
e7q_joint_binding_not_yet_viable
e7q_artifact_or_task_too_easy
```

## Checker Gates

The checker fails on missing artifacts, missing system variants, missing row-level samples, missing training rows, missing reuse rows, failed deterministic replay, replay hash mismatch, hidden-answer leakage flags, dense graph marked as primary, or claims outside this controlled numeric proxy.

## Boundary

E7Q is a controlled numeric Flow[D] router-pocket binding probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
