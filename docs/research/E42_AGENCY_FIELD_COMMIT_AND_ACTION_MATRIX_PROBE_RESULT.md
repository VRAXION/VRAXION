# E42 Agency Field Commit And Action Matrix Probe Result

Decision:

```text
e42_agency_field_positive
```

Run root:

```text
target/pilot_wave/e42_agency_field_commit_and_action_matrix_probe
```

Artifact sample:

```text
docs/research/artifact_samples/e42_agency_field_commit_and_action_matrix_probe
```

## Primary Result

The full Agency Field learned a small ALU/Logic-Atom action matrix over
mechanical Flow, Ground, Proposal, Trace, and Cost views. It selected the
correct action and trace reason on all primary rows while the direct and simple
arbiter controls remained weak.

| System | Action | Trace | Wrong Commit | Missed Commit | Ask | Defer |
|---|---:|---:|---:|---:|---:|---:|
| `agency_field_full_views_grow_shrink` | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| `agency_field_without_ground` | 0.900000 | 0.600000 | 0.000000 | 0.100000 | 1.000000 | 1.000000 |
| `simple_priority_arbiter` | 0.600000 | 0.600000 | 0.300000 | 0.000000 | 1.000000 | 0.000000 |
| `direct_pocket_action_baseline` | 0.400000 | 0.400000 | 0.500000 | 0.000000 | 0.000000 | 0.000000 |
| `random_action_control` | 0.151042 | 0.151042 | 0.310417 | 0.172917 | 0.156250 | 0.135417 |

The learned primary genome used 8 atoms and 13 total conditions. The useful
shape was not direct pocket overwrite. It was:

```text
Pocket / Logic Atom proposals -> Proposal Memory
Flow + Ground + Trace + Cost views -> Agency Field
Agency Field -> COMMIT / REJECT / DEFER / ASK / CALL / ANSWER
```

## Mutation Backend

```text
accepted_mutations = 213
rejected_mutations = 2347
rollback_count = 2347
parameter_hash = e3f33b9c50d29f0bc745423b634a4e61bf72ff30190b31ccfadb4f882cef9c31
```

The final mutation fitness repaired trace failures as well as wrong actions.
Noise/decoy conditions were allowed but penalized, which prevented the earlier
failure mode where a valid CALL atom accidentally depended on a random decoy
bit.

## Confirm Seeds

Four parallel confirm runs also passed:

| Seed | Decision | Full Action | Full Trace | Wrong Commit | Checker |
|---:|---|---:|---:|---:|---|
| 42022 | `e42_agency_field_positive` | 1.000000 | 1.000000 | 0.000000 | pass |
| 42023 | `e42_agency_field_positive` | 1.000000 | 1.000000 | 0.000000 | pass |
| 42024 | `e42_agency_field_positive` | 1.000000 | 1.000000 | 0.000000 | pass |
| 42025 | `e42_agency_field_positive` | 1.000000 | 1.000000 | 0.000000 | pass |

## Checker

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_hash_match = true
```

## Interpretation

E42 supports the minimal Agency Field idea on a controlled symbolic/numeric
proxy:

```text
Pocket operators propose.
Flow Field carries active state.
Ground Field anchors contradiction/stability.
Trace Ledger supplies evidence status.
Agency Field decides what becomes action or committed state.
```

The result does not say that a central monolithic brain should solve the task.
The monolith/oracle controls are invalid references. The useful evidence is
that the small grow/shrink Agency genome matched the reference decision
behavior, while direct pocket write and simple priority arbitration failed, and
the no-Ground ablation lost action and trace reliability.

## Boundary

E42 is a controlled symbolic/numeric Agency Field proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.
