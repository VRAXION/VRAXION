# E57 Output Egress Field Multi-Resolution Renderer Probe Result

Status: completed and checker validated.

## Decision

```text
decision = e57_multi_resolution_egress_renderer_confirmed
checker_failure_count = 0
sample_only_checker_passed = true
run_id = c9cf7513a1881079
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Overall Systems

| system | success | mode accuracy | multires write | false output | stale leak | mean cost | net utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_only_output | 0.250000 | 0.250000 | 0.000000 | 0.750000 | 0.000000 | 0.250 | -0.950000 |
| short_text_only_output | 0.163327 | 0.163327 | 0.000000 | 0.836673 | 0.000000 | 1.000 | -1.175351 |
| long_text_only_output | 0.375000 | 0.211673 | 0.000000 | 0.625000 | 0.000000 | 2.600 | -0.775230 |
| direct_pocket_to_text_unsafe | 0.625000 | 0.625000 | 0.000000 | 0.375000 | 0.250000 | 1.674 | -0.180572 |
| naive_length_egress_router | 0.625000 | 0.625000 | 0.000000 | 0.375000 | 0.000000 | 1.702 | -0.019801 |
| agency_committed_single_resolution | 0.875000 | 0.875000 | 0.000000 | 0.125000 | 0.000000 | 1.151 | 0.648066 |
| agency_committed_multi_resolution_renderer | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.201 | 0.970566 |
| oracle_egress_reference | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.201 | 0.970566 |
| random_output_control | 0.357178 | 0.199445 | 0.204982 | 0.491769 | 0.000000 | 1.408 | -0.616375 |

## Multi-Resolution Focus

| system | multires write success | trace-backed output | stale proposal leak | net utility |
|---|---:|---:|---:|---:|
| compact_only_output | 0.000000 | 0.250000 | 0.000000 | -0.950000 |
| short_text_only_output | 0.000000 | 0.163327 | 0.000000 | -1.175351 |
| long_text_only_output | 0.000000 | 0.375000 | 0.000000 | -0.775230 |
| direct_pocket_to_text_unsafe | 0.000000 | 0.000000 | 0.250000 | -0.180572 |
| naive_length_egress_router | 0.000000 | 0.625000 | 0.000000 | -0.019801 |
| agency_committed_single_resolution | 0.000000 | 0.875000 | 0.000000 | 0.648066 |
| agency_committed_multi_resolution_renderer | 1.000000 | 1.000000 | 0.000000 | 0.970566 |
| oracle_egress_reference | 1.000000 | 1.000000 | 0.000000 | 0.970566 |
| random_output_control | 0.204982 | 0.357178 | 0.000000 | -0.616375 |

## Recommendation

```text
recommended_policy = agency_committed_multi_resolution_renderer
compact_action = 1x32 byte action field
short_text = 1x256 byte output field
long_text = 4x256 byte output field
multi_resolution = compact + short + long/detail output fields
lock_statement = Render output only from Agency-committed state. Use compact, short, long, or multi-resolution Egress Field modes as needed; never render final output directly from raw Pocket proposals.
```

## Interpretation

The output path should mirror the input path structurally, but not permissively.
Output is rendered from Agency-committed Flow/Ground/Trace state, not directly
from raw Pocket proposals. Multi-resolution output matters when a decision must
simultaneously provide a compact action, a short answer surface, and a detailed
trace-backed byte/text form.

Direct Pocket-to-text output is an unsafe control because stale or unresolved
proposal content can leak into final output. A single-resolution committed
renderer remains useful for simple rows, but it cannot satisfy rows that require
compact and detailed output to agree.

## Boundary

E57 is a deterministic output/egress field probe. It tests whether
Agency-committed state can be rendered into compact, short-text, long-text, and
multi-resolution byte/text output fields without direct proposal leakage. It
does not claim raw language reasoning, AGI, consciousness, deployment quality,
or model-scale behavior.
