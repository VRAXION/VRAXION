# E41 Logic Atom Genome Grow/Shrink And Commit Probe Result

Decision:

```text
e41_logic_atom_grow_shrink_commit_positive
```

Run root:

```text
target/pilot_wave/e41_logic_atom_genome_grow_shrink_and_commit_probe
```

Artifact sample:

```text
docs/research/artifact_samples/e41_logic_atom_genome_grow_shrink_and_commit_probe
```

## Result

E41 confirms the next step after E40 in this controlled spatial Flow-grid
proxy: small Logic Atoms work cleanly as proposal generators when an Arbiter
commit layer decides WRITE/REJECT/DEFER. A grow/shrink mutation genome learned a
compact proposal program from one unconditional WRITE atom.

Primary run:

| System | Exact | Action accuracy | False commit | Missed commit |
|---|---:|---:|---:|---:|
| `grow_shrink_logic_atom_genome` | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| `fixed_slot_proposal_arbiter` | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| `direct_write_logic_atom_baseline` | 0.544271 | 0.528646 | 0.471354 | 0.000000 |
| `proposal_without_arbiter` | 0.544271 | 0.528646 | 0.471354 | 0.000000 |
| `random_genome_control` | 0.486979 | 0.333333 | 0.170573 | 0.352865 |
| `full_flow_painter_control` | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

The full-flow painter is a diagnostic invalid control: it succeeds by writing
the whole grid target and has `write_spread_ratio = 1.0`. The learned
grow/shrink genome uses sparse patch writes with `write_spread_ratio =
0.038411`.

## Learned Genome

The primary learned genome:

```text
atom_0:
  IF missing_is_0
  THEN WRITE

atom_1:
  IF blocker_is_1 AND missing_is_0
  THEN REJECT

otherwise:
  no proposal -> Arbiter DEFER
```

This is a useful shape: the genome did not need a separate explicit DEFER rule.
Unresolved state is represented by no valid proposal, and the Arbiter safely
defers.

Mutation stats:

```text
accepted_mutations = 200
rejected_mutations = 1816
rollback_count     = 1816
initial_score      = 0.583938
final_score        = 0.995800
```

## Multi-Seed Confirm

Four independent confirm seeds also reached the positive decision and passed
the checker:

| Seed | Decision | Grow exact | Grow action | Direct exact | Random action |
|---:|---|---:|---:|---:|---:|
| 41022 | `e41_logic_atom_grow_shrink_commit_positive` | 1.000000 | 1.000000 | 0.557292 | 0.325521 |
| 41023 | `e41_logic_atom_grow_shrink_commit_positive` | 1.000000 | 1.000000 | 0.528646 | 0.299479 |
| 41024 | `e41_logic_atom_grow_shrink_commit_positive` | 1.000000 | 1.000000 | 0.572917 | 0.330729 |
| 41025 | `e41_logic_atom_grow_shrink_commit_positive` | 1.000000 | 1.000000 | 0.562500 | 0.359375 |

## Checker

Target checker:

```text
passed = true
failure_count = 0
```

Sample-only checker:

```text
passed = true
failure_count = 0
```

Deterministic replay:

```text
passed = true
deterministic_replay_match_rate = 1.0
```

## Interpretation

E41 supports this local architectural rule:

```text
Logic Atom = IF conditions THEN proposal
Arbiter    = commit/reject/defer proposals
Flow write = happens only after commit
```

This is better than direct overwrite for the tested proxy because it separates
candidate transforms from stable Flow updates.

Boundary: E41 is a controlled spatial Flow-grid proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.
