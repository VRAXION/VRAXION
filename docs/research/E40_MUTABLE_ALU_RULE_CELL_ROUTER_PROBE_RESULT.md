# E40 Mutable ALU Rule Cell Router Probe Result

Status: complete.

Decision:

```text
e40_mutable_alu_rule_cell_router_positive
```

Run root:

```text
target/pilot_wave/e40_mutable_alu_rule_cell_router_probe
```

Artifact sample:

```text
docs/research/artifact_samples/e40_mutable_alu_rule_cell_router_probe
```

Checker:

```text
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
deterministic_replay_passed = true
```

## Result

Official evidence run:

| System | Exact | Cell accuracy | Read spread | Write spread | Scan cells |
|---|---:|---:|---:|---:|---:|
| `oracle_alu_rule_reference` | 1.000000 | 1.000000 | 0.069043 | 0.069043 | 0.0 |
| `flat_marker_table_router` | 0.287500 | 0.970416 | 0.329242 | 0.062500 | 72.0 |
| `location_only_fixed_call_router` | 0.287500 | 0.970416 | 0.329242 | 0.062500 | 72.0 |
| `boolean_alu_without_op_router` | 0.496875 | 0.976703 | 0.335785 | 0.069043 | 72.0 |
| `mutable_alu_rule_cell_router` | 1.000000 | 1.000000 | 0.335785 | 0.069043 | 72.0 |
| `scan_all_rule_control` | 1.000000 | 1.000000 | 0.775165 | 0.069043 | 210.6 |
| `full_flow_painter_control` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 256.0 |
| `random_rule_control` | 0.012500 | 0.937061 | 0.092334 | 0.076709 | 0.0 |

Multi-seed confirm:

| Run | Decision | ALU exact | Flat exact | Scale-only exact | ALU write spread |
|---|---|---:|---:|---:|---:|
| primary `40021` | `e40_mutable_alu_rule_cell_router_positive` | 1.000000 | 0.287500 | 0.496875 | 0.069043 |
| confirm `40022` | `e40_mutable_alu_rule_cell_router_positive` | 1.000000 | 0.259766 | 0.500000 | 0.066498 |
| confirm `40023` | `e40_mutable_alu_rule_cell_router_positive` | 1.000000 | 0.244141 | 0.496094 | 0.072815 |
| confirm `40024` | `e40_mutable_alu_rule_cell_router_positive` | 1.000000 | 0.283203 | 0.507812 | 0.069427 |

## Learned Logic Atom Program

The primary mutable ALU/Logic-Atom router started from:

```text
all scale rules output 4
all op rules output copy
```

and learned:

```text
IF local[+1,0] == 0 AND local[0,+1] == 0 THEN scale = 2
IF local[+1,0] == 1 AND local[0,+1] == 0 THEN scale = 4
IF local[+1,0] == 0 AND local[0,+1] == 1 THEN scale = 6
IF local[+1,0] == 1 AND local[0,+1] == 1 THEN scale = 4

IF local[+2,0] == 0 THEN op = invert
IF local[+2,0] == 1 THEN op = threshold
```

Mutation stats:

```text
accepted = 1745
rejected = 943
rollback = 943
```

## Interpretation

E40 supports the core idea:

```text
Logic Atom / ALU-rule cells are a mutation-friendly Pocket genome for small
control/routing/commit transformations.
```

The key separation:

```text
flat marker table:
  can find the header but cannot condition on multiple cells.

boolean scale-only ALU:
  learns location/scale but fails because op also depends on a condition cell.

mutable ALU rule-cell router:
  learns scale and op rules from visible Flow cells with small write footprint.

scan-all/full-flow controls:
  solve only by excessive scan/read or diffuse write.
```

This makes the ALU layer a strong candidate for:

```text
router decisions
commit guards
evidence acceptance
requested-feature matching
small local Flow writes
```

## Caveat

This is not yet full self-growing program synthesis. The rule slots and relative
condition cells were scaffolded; mutation learned the outputs and rule behavior
inside that scaffold. The next harder test should allow rule growth/shrink,
relative input mutation, and proposal/commit semantics.

Boundary: E40 is a controlled spatial Flow-grid proxy. It does not prove raw
language reasoning, AGI, consciousness, deployed-model behavior, or model-scale
behavior.
