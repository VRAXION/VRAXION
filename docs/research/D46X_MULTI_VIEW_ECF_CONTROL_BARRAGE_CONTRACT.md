# D46X Multi-View ECF Control Barrage Contract

D46X tests whether the ECF control stack benefits from multiple independent views over the same controlled symbolic IPF task.

Scope:

```text
controlled symbolic primitive discovery only
IPF/ECF support-policy control only
candidate / family / equivalence metrics separated
no raw visual Raven claim
no AGI / consciousness / architecture superiority claim
```

Views under test:

```text
scalar score
vector field
entropy
margin
collision
support independence
counterfactual top1-vs-top2
```

Arms:

```text
SCALAR_ARGMAX_ONLY
VECTOR_FIELD_ONLY
ENTROPY_SUPPORT_POLICY
MARGIN_SUPPORT_POLICY
COLLISION_SUPPORT_POLICY
SUPPORT_INDEPENDENCE_DEDUP_POLICY
COUNTERFACTUAL_TOP1_TOP2_POLICY
SCALAR+VECTOR
VECTOR+ENTROPY+MARGIN
VECTOR+COLLISION+INDEPENDENCE
FULL_MULTI_VIEW_ECF_POLICY
RANDOM_EXTRA_SUPPORT_CONTROL
BAD_VIEW_CONTROL
SHUFFLED_VECTOR_FIELD_CONTROL
NO_COUNTERFACTUAL_CONTROL
```

Primary positive gate:

```text
FULL_MULTI_VIEW_ECF_POLICY beats all single-view arms and controls
clean >= 0.995
correlated >= 0.95
adversarial >= 0.95
clean regression <= 0.005
support cost <= D46 robust combined support cost
candidate/family/equivalence metrics separated
controls included and worse
failed jobs visible
```

Decision routes:

```text
multi_view_ecf_control_barrage_positive -> D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT
multi_view_ecf_positive_high_cost -> D46Y_VIEW_COST_OPTIMIZATION
single_view_dominates_multiview_redundant -> D47_WITH_DOMINANT_VIEW
multiview_ecf_not_robust -> D46_REPAIR_SUPPORT_ROBUSTNESS
```

Boundary: D46X only tests multi-view ECF control in controlled symbolic primitive discovery. It does not prove raw visual Raven reasoning, Raven solved, AGI, consciousness, architecture superiority, or that intelligence is literally a force.
