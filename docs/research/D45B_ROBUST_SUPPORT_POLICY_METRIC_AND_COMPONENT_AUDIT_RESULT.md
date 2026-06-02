# D45B Robust Support Policy Metric And Component Audit Result

D45B completed locally on branch:

```text
codex/d45b-robust-support-policy-audit
```

Artifact root:

```text
target/pilot_wave/d45b_robust_support_policy_metric_and_component_audit/smoke/
```

Decision:

```text
decision = robust_policy_components_identified
verdict = D45B_ROBUST_COMPONENTS_IDENTIFIED
next = D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRM
```

Primary `ALL28_UNORDERED` component audit:

```text
NAIVE_IPF_BASELINE:
  clean = 0.9996
  correlated = 0.0000
  adversarial = 0.6905
  total_support = 5.000
  counter_support = 0.000

COUNTER_SUPPORT_ONLY:
  clean = 1.0000
  correlated = 0.9972
  adversarial = 0.9999
  total_support = 6.863
  counter_support = 1.863

DEDUP_PLUS_COUNTER_SUPPORT:
  clean = 1.0000
  correlated = 0.9962
  adversarial = 1.0000
  total_support = 6.863
  counter_support = 1.863

FULL_ROBUST_COMBINED_REPLAY:
  clean = 1.0000
  correlated = 0.9974
  adversarial = 0.9999
  total_support = 6.863
  counter_support = 1.863

ROBUST_COMBINED_COST_CAPPED_7:
  clean = 1.0000
  correlated = 0.9673
  adversarial = 0.9981
  total_support = 6.242
  counter_support = 1.242

RANDOM_EXTRA_SUPPORT_CONTROL:
  correlated = 0.6142

BAD_ROBUSTNESS_SIGNAL_CONTROL:
  correlated = 0.0907

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  correlated = 0.1595
  adversarial = 0.3536
```

Metric semantics audit result:

```text
D45 average_support_used meant total support consumed.
D45B separates original_support_used, counter_support_used, and total_support_used.
counter_support_resolution_rate is not final accuracy; it only counts rows where a requested counter-support fixed a row that naive had wrong.
```

Interpretation:

```text
The repair is not explained by random extra evidence.
The main repair component is targeted counter-support.
Dedup-only does not solve correlated support.
Shuffled counter-support is much worse than real counter-support.
Cost cap 7 retains strong repair with lower support cost.
```

Boundary: controlled symbolic IPF/ECF robust support policy audit only; no raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or literal-force claim.
