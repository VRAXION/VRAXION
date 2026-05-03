# Phase D29 - Route-Selected Execution

## Verdict

```text
D29_ROUTE_EXECUTION_PASS
```

D29 connects the D28 C-router to the existing D-worker probes and verifies that
inactive lanes stay empty:

```text
bytes -> A/B codec -> B64 -> C-router -> selected D worker -> output
                              |
                              '-- inactive lanes must remain empty
```

## Confirm Result

Run shape:

```text
mode: confirm
samples_per_class: 8,192
episodes: 8,192
control_repeats: 8
sample_count: 57,353
artifact: tools/ab_window_codec_v1.json
```

Result:

```text
route_acc:                  100%
selected_output_acc:        100%
inactive_lanes_empty_acc:   100%
executable_worker_acc:      100%
policy_acc:                 100%
unsupported_alu_ok:         true

LANG / ALU / MEM / TRANSFORM / UNKNOWN:
  route_acc: 100%
  exact_acc: 100%

wrong_route_control:
  selected_output_acc: 0%

random_route_controls:
  selected_output_acc: about 19.8% to 20.4%
```

Five balanced routes make random-route chance about 20%.

## Integration Smoke

```text
REV ABC  -> TRANSFORM -> CBA
ROT XYZ  -> TRANSFORM -> YZX
1+2      -> ALU       -> 3
99-4     -> ALU       -> 95
STORE X  -> MEM       -> STORED:X
QUERY X  -> MEM       -> X
THE CAT  -> LANG      -> NO_LANG_WORKER
THE+CAT  -> UNKNOWN   -> REJECT
27*852   -> ALU       -> UNSUPPORTED_ALU_OP
```

The lane table confirms only the selected lane emits output.

## Interpretation

The current block line is now:

```text
A = byte codec
B = B64 data bus
C = content router / switchboard
D = worker blocks
    D24 transform
    D25 memory
    D27 ALU
```

D28 proved route selection. D29 proves route-selected execution:

```text
route decision -> selected worker runs -> output returns -> inactive lanes empty
```

This is still a probe/reference system, not a learned general AI. It is the
first complete ABCD switchboard proof over the current AB/B64 bus.

## Scope Boundary

`*` routes to ALU but is intentionally not executed:

```text
27*852 -> ALU -> UNSUPPORTED_ALU_OP
```

This is correct for D29A because D27 does not implement multiplication.

`LANG` routes correctly, but no language worker exists yet:

```text
THE CAT -> LANG -> NO_LANG_WORKER
```

## Next Step

```text
D30: composed command episodes
```

Examples:

```text
STORE X
QUERY REV X
ALU result -> memory
memory result -> transform
```

## Artifacts

Generated outputs:

```text
output/phase_d29_route_selected_execution_20260503/smoke/
output/phase_d29_route_selected_execution_20260503/main/
output/phase_d29_route_selected_execution_20260503/confirm/
output/phase_d29_route_selected_execution_20260503/integration_smoke/
```

Tracked implementation:

```text
tools/_scratch/d29_route_selected_execution_probe.py
```
