# D46 Robust Support Policy Scale Confirm Contract

D46 scale-confirms the robust support policy after D45B identified the working components.

Scope: controlled symbolic IPF/ECF support policy only.

Policies:

```text
NAIVE_IPF_BASELINE
COUNTER_SUPPORT_ONLY
FULL_ROBUST_COMBINED_REPLAY
ROBUST_COMBINED_COST_CAPPED_5
ROBUST_COMBINED_COST_CAPPED_7
ROBUST_COMBINED_COST_CAPPED_9
RANDOM_EXTRA_SUPPORT_CONTROL
BAD_ROBUSTNESS_SIGNAL_CONTROL
SHUFFLED_COUNTER_SUPPORT_CONTROL
```

Positive gate:

```text
clean >= 0.995
correlated >= 0.95
adversarial >= 0.95
mixed >= 0.95
min seed correlated/adversarial >= 0.90
clean regression <= 0.005
controls worse
failed jobs = 0
support cost reported
```

Boundary: no raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or literal-force claim.
