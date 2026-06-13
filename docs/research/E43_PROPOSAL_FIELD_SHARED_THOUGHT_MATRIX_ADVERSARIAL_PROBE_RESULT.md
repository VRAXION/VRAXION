# E43 Proposal Field Shared Thought Matrix Adversarial Probe Result

Decision:

```text
e43_shared_proposal_field_adversarial_confirmed
```

Run root:

```text
target/pilot_wave/e43_proposal_field_shared_thought_matrix_adversarial_probe
```

Artifact sample:

```text
docs/research/artifact_samples/e43_proposal_field_shared_thought_matrix_adversarial_probe
```

## Primary Result

E43 stress-tested the Proposal Field / Thought Field idea against collision,
poisoning, flooding, stale replay, Ground conflicts, Trace mismatch,
location/scale poison, partial truths, missing valid proposals, and colluding
wrong pockets.

The important result is not that a shared field alone works. It does not. The
confirmed pattern is:

```text
shared Proposal Field + Agency commit boundary
```

| System | Success | Action | False Commit | Toxic Commit | Stale Commit | Collision | No-valid Defer | Illegal Direct Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_proposal_field_plus_agency` | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| `per_pocket_planes_plus_agency` | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| `shared_proposal_field` | 0.000000 | 0.090909 | 0.909091 | 0.545455 | 0.090909 | 0.000000 | 0.000000 | 0.000000 |
| `per_pocket_proposal_planes` | 0.000000 | 0.090909 | 0.909091 | 0.545455 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `explicit_single_proposal_packet` | 0.000000 | 0.363636 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `direct_flow_write_baseline` | 0.000000 | 0.090909 | 0.909091 | 0.545455 | 0.090909 | 0.000000 | 0.000000 | 1.000000 |
| `toxic_pocket_control` | 0.000000 | 0.090909 | 0.909091 | 0.545455 | 0.090909 | 0.000000 | 0.000000 | 0.000000 |
| `proposal_flood_control` | 0.000000 | 0.000000 | 1.000000 | 0.636364 | 0.090909 | 0.000000 | 0.000000 | 0.000000 |
| `stale_proposal_control` | 0.000000 | 0.090909 | 0.909091 | 0.545455 | 0.090909 | 0.000000 | 0.000000 | 0.000000 |

The invalid `oracle_commit_reference` reached 1.000000 and is used only as a
ceiling control.

## Shared Vs Per-Pocket Planes

```text
shared_agency_success      = 1.000000
per_pocket_agency_success  = 1.000000
per_pocket_plane_gain      = 0.000000
shared_field_collision_rate = 0.000000
```

Per-pocket planes did not beat shared Proposal Field when the shared field kept
mechanical cycle, Trace, Ground, evidence, and target/value compatibility
information available to Agency. Therefore E43 does not require per-pocket
proposal planes yet.

## Adversarial Interpretation

The raw shared Proposal Field is unsafe:

```text
false_commit_rate = 0.909091
toxic_commit_rate = 0.545455
stale_commit_rate = 0.090909
```

The safe pattern is:

```text
Pocket writes temporary proposal into Proposal Field.
Proposal Field is not truth.
Agency checks cycle + Trace + Ground + evidence + target/value compatibility.
Only then can Flow be updated.
Proposal Field clears after the decision cycle.
```

So the canonical lock is not:

```text
shared Proposal Field alone
```

It is:

```text
shared Proposal Field + Agency commit boundary + cycle/trace/ground/evidence isolation
```

## Confirm Seeds

Four confirm runs also passed:

| Seed | Decision | Shared+Agency Success | Per-Pocket Gain | Checker |
|---:|---|---:|---:|---|
| 43022 | `e43_shared_proposal_field_adversarial_confirmed` | 1.000000 | 0.000000 | pass |
| 43023 | `e43_shared_proposal_field_adversarial_confirmed` | 1.000000 | 0.000000 | pass |
| 43024 | `e43_shared_proposal_field_adversarial_confirmed` | 1.000000 | 0.000000 | pass |
| 43025 | `e43_shared_proposal_field_adversarial_confirmed` | 1.000000 | 0.000000 | pass |

## Checker

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_hash_match = true
```

## Boundary

E43 is a controlled symbolic/numeric Proposal Field and Agency Field proxy. It
does not prove raw language reasoning, AGI, consciousness, deployed-model
behavior, or model-scale behavior.
