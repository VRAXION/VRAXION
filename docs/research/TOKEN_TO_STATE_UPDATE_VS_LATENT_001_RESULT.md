# TOKEN_TO_STATE_UPDATE_VS_LATENT_001 Result

## Goal

Test explicit ledger, latent recurrent state, and hybrid state supervision on a controlled object-lifecycle counting suite.

## Arm Summary

| Arm | Answer | Same-token | Order-shuffle | Coref | Invalid restore | Linear count probe | Lifecycle probe |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAG_OF_TOKENS_MLP` | `0.695` | `0.500` | `0.500` | `0.691` | `0.714` | `nan` | `nan` |
| `EXPLICIT_LEDGER_ORACLE` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `nan` | `nan` |
| `HYBRID_STATE_TEACHER` | `0.786` | `0.900` | `0.900` | `0.765` | `0.795` | `0.788` | `0.816` |
| `LATENT_GRU_ANSWER_ONLY` | `0.799` | `0.850` | `0.850` | `0.784` | `0.814` | `0.790` | `0.678` |
| `LATENT_GRU_FROZEN_LINEAR_PROBES` | `0.799` | `0.850` | `0.850` | `0.784` | `0.814` | `0.790` | `0.678` |
| `SHUFFLED_STATE_TEACHER` | `0.795` | `0.750` | `0.750` | `0.781` | `0.809` | `0.790` | `0.681` |
| `STATIC_POSITION_MLP` | `0.640` | `0.600` | `0.600` | `0.618` | `0.646` | `nan` | `nan` |

## Verdict

```json
[
  "EXPLICIT_LEDGER_REQUIRED_FOR_NOW"
]
```

## Interpretation

The stress is valid enough for this first gate: `EXPLICIT_LEDGER_ORACLE` passes, while `BAG_OF_TOKENS_MLP` fails the same-token/order controls at `0.500`. This means the adversarial pairs are doing useful work instead of letting a bag-of-events shortcut pass.

The latent GRU learns a partial story-counting behavior (`0.799` answer accuracy), but it does not satisfy the state-positive gates. Its frozen linear count probe is only `0.790`, and lifecycle probe accuracy is `0.678`, below the locked thresholds. That makes the result an explicit negative for reliable latent state tracking, not a proof that a latent recurrent state is enough.

`HYBRID_STATE_TEACHER` improves lifecycle probe accuracy (`0.816`) but does not improve final answer accuracy over the answer-only GRU, and it does not separate cleanly from the shuffled control. That means the auxiliary state targets are not yet a causal win in this carrier/setup.

The current conclusion is:

```text
explicit ledger/state scaffolding is still required for this toy gate,
or the latent carrier/training setup must be strengthened before we can remove it.
```

Implementation note: the corrected run uses the GRU state at the last real token, not after padding. The earlier padding-tail collapse was fixed before this result was recorded.

## Claim Boundary

Controlled toy grammar only. The result does not prove open-ended natural-language grounding or consciousness.
