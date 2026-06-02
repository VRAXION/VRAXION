# D46 Robust Support Policy Scale Confirm Result

D46 scale-lite completed locally on branch:

```text
codex/d45b-robust-support-policy-audit
```

Artifact root:

```text
target/pilot_wave/d46_robust_support_policy_scale_confirm/smoke/
```

Decision:

```text
decision = robust_support_policy_scale_confirmed
verdict = D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRMED
next = D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT
```

Primary `ALL28_UNORDERED` metrics:

```text
NAIVE_IPF_BASELINE:
  clean = 0.9994
  correlated = 0.0000
  adversarial = 0.6946
  mixed = 0.9811
  min_seed_correlated = 0.0000
  min_seed_adversarial = 0.6850
  support = 5.000

COUNTER_SUPPORT_ONLY:
  clean = 1.0000
  correlated = 0.9959
  adversarial = 0.9998
  mixed = 1.0000
  min_seed_correlated = 0.9931
  min_seed_adversarial = 0.9988
  support = 6.864

FULL_ROBUST_COMBINED_REPLAY:
  clean = 1.0000
  correlated = 0.9974
  adversarial = 0.9998
  mixed = 1.0000
  min_seed_correlated = 0.9962
  min_seed_adversarial = 0.9994
  support = 6.864

ROBUST_COMBINED_COST_CAPPED_7:
  clean = 1.0000
  correlated = 0.9673
  adversarial = 0.9982
  mixed = 0.9998
  min_seed_correlated = 0.9625
  min_seed_adversarial = 0.9975
  support = 6.242

RANDOM_EXTRA_SUPPORT_CONTROL:
  correlated = 0.6092

BAD_ROBUSTNESS_SIGNAL_CONTROL:
  correlated = 0.0912

SHUFFLED_COUNTER_SUPPORT_CONTROL:
  correlated = 0.1604
  adversarial = 0.3464

clean_regression_vs_naive = -0.000625
```

Interpretation:

```text
The D45/D45B robust support policy scale-confirmed under scale-lite settings.
Targeted counter-support remains the main useful repair path.
Random extra evidence, bad robustness signal, and shuffled counter-support remain much worse.
Cost cap 7 remains viable but slightly below full robust correlated accuracy.
```

Boundary: controlled symbolic IPF/ECF robust support scale confirm only; no raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or literal-force claim.
