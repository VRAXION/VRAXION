# D46X Multi-View ECF Control Barrage Result

Status:

```text
completed
```

Branch:

```text
codex/d46x-multi-view-ecf-control-barrage
```

Artifact root:

```text
target/pilot_wave/d46x_multi_view_ecf_control_barrage/smoke/
```

Decision:

```text
decision = multi_view_ecf_control_barrage_positive
verdict = D46X_MULTI_VIEW_ECF_CONTROL_BARRAGE_POSITIVE
next = D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT
rows_evaluated = 3660000
```

Primary `ALL28_UNORDERED`, support budget 5:

```text
SCALAR_ARGMAX_ONLY:
  clean = 0.9995
  correlated = 0.0000
  adversarial = 0.6917
  mixed = 0.9802
  support = 5.000

COUNTERFACTUAL_TOP1_TOP2_POLICY:
  clean = 1.0000
  correlated = 0.7398
  adversarial = 0.9802
  mixed = 0.9991
  support = 5.621

FULL_MULTI_VIEW_ECF_POLICY:
  clean = 1.0000
  correlated = 0.9683
  adversarial = 0.9985
  mixed = 0.9999
  support = 6.243

RANDOM_EXTRA_SUPPORT_CONTROL:
  correlated = 0.6264
  adversarial = 0.9619
  support = 5.621

BAD_VIEW_CONTROL:
  correlated = 0.0875
  adversarial = 0.0295

SHUFFLED_VECTOR_FIELD_CONTROL:
  correlated = 0.0000
  adversarial = 0.0000

NO_COUNTERFACTUAL_CONTROL:
  correlated = 0.0000
  adversarial = 0.1310
```

Interpretation:

```text
The full multi-view policy beat the single-view arms and controls.
The useful single view was counterfactual top1-vs-top2, but it did not repair correlated support enough alone.
The vector-only view was not useful by itself in this setup.
The full stack improved correlated robustness over random extra support while staying below D46 robust combined support cost.
```

Boundary: D46X only tests multi-view ECF control in controlled symbolic primitive discovery. It does not prove raw visual Raven reasoning, Raven solved, AGI, consciousness, architecture superiority, or that intelligence is literally a force.
