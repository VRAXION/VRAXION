# Phase D28 - Content-Based C-Router

## Verdict

```text
D28_CONTENT_ROUTER_PASS
```

D28 adds the first content-based C-router probe on top of the frozen AB/B64
surface:

```text
bytes -> A/B codec -> B64 -> C-router -> route label
```

It is router-only. It predicts where an input should go; it does not execute the
worker blocks yet.

## Route Labels

```text
LANG       natural text-like windows, for example THE CAT
ALU        arithmetic expressions, for example 1+888
MEM        memory commands, for example STORE X
TRANSFORM  transform commands, for example REV ABC
UNKNOWN    ambiguous/noisy/unsupported windows
```

Integration smoke:

```text
1+888    -> ALU
REV ABC  -> TRANSFORM
STORE X  -> MEM
THE CAT  -> LANG
THE+CAT  -> UNKNOWN
```

## Main Confirm

Run shape:

```text
mode: confirm
samples_per_class: 8,192
heldout_per_class: 4,096
control_repeats: 8
artifact: tools/ab_window_codec_v1.json
```

Result:

```text
train:
  route_acc: 100%
  every class: 100%

heldout:
  route_acc: 100%
  every class: 100%

adversarial:
  route_acc: 100%
  every class: 100%

shuffled_label controls:
  route_acc: about 19.4% to 20.1%

random_projection controls:
  route_acc: about 19.6% to 20.3%
```

Since there are five balanced route labels, the failed controls are near chance.

## Interpretation

The current block layout is now:

```text
A = byte codec
B = 8-byte/B64 data bus
C = content router/controller
D = worker blocks
    D24 transform
    D25 memory
    D27 ALU
```

D28 proves that the B64 bus carries enough surface information to route content
into the right high-level worker lane:

```text
THE CAT -> language lane
27*852  -> ALU lane
STORE X -> memory lane
REV ABC -> transform lane
```

This is not yet a release AI. It is the missing routing skeleton between the
AB codec and existing worker probes.

## Caveat

D28 v1 uses a compact crystallized/sparse predicate route head over the B64
surface. It is not yet an emergent learned semantic router. The purpose of this
phase is to prove clean route separability and adversarial controls before
composing worker execution.

## Next Step

```text
D29: route-selected execution
```

D29 should connect the D28 route decision to the existing D workers:

```text
ALU        -> D27
TRANSFORM  -> D24
MEM        -> D25
LANG       -> label only until a language worker exists
UNKNOWN    -> reject / no-op
```

## Artifacts

Generated outputs:

```text
output/phase_d28_content_router_20260503/smoke/
output/phase_d28_content_router_20260503/main/
output/phase_d28_content_router_20260503/confirm/
output/phase_d28_content_router_20260503/integration_smoke/
```

Tracked implementation:

```text
tools/_scratch/d28_content_router_probe.py
```
