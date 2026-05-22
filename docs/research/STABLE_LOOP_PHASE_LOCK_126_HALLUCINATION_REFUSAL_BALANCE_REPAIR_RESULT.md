# STABLE_LOOP_PHASE_LOCK_126_HALLUCINATION_REFUSAL_BALANCE_REPAIR_RESULT

## Result Contract

The 126 runner writes the result under:

```text
target/pilot_wave/stable_loop_phase_lock_126_hallucination_refusal_balance_repair/
```

The expected positive decision is:

```text
decision = hallucination_refusal_balance_repair_success
next = 127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM
```

Positive 126 requires `HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE` plus
evidence that hallucination/refusal calibration improved without refusal-only
training.

## Required Reports

The result includes:

- calibration repair metrics
- answerable-vs-refusal report
- always-refuse degeneration report
- reasoning/state preservation report
- retention report
- collapse metrics
- namespace audit
- leakage audit
- control arm report
- human-readable samples
- failure case samples

The runner writes partial progress throughout the run, including startup,
upstream verification, dataset build, leakage audit, seed training start,
training heartbeat, rollout eval heartbeat, seed final eval, aggregate analysis,
decision writing, and final verdict.

## Boundary

126 is targeted research repair only. It is not GPT-like assistant readiness,
not open-domain assistant readiness, not production chat, not public API, not
deployment readiness, not safety alignment, and not Hungarian assistant
readiness.

