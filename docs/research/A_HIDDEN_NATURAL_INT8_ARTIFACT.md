# A-HiddenNatural Int8 Artifact

Status: `A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS`

This phase converts the current `A-HiddenNaturalMarginPolish` candidate from
float JSON search output into an explicit native integer artifact.

## Artifact

- Path: `tools/a_hidden_natural_margin_int8_v1.json`
- Storage: `int8_q6`
- Scale: `64`
- Formula: `weight = q / 64`
- Shape: `8 visible bits -> 8 hidden -> 16 A lanes`
- Decoder: tied transpose chain, no independent decoder weights.

## Verification

Verifier:

```powershell
python tools\a_hidden_natural_int8_artifact.py --verify-artifact tools\a_hidden_natural_margin_int8_v1.json
```

Result:

```text
verdict: A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS
exact_byte_acc: 1.0
bit_acc: 1.0
byte_margin_min: 3.515625
hidden_collisions: 0
hidden_in_edge_count: 8
hidden_out_edge_count: 30
```

## Important Distinction

The search outputs were stored as JSON decimal values because the evaluator used
NumPy matrices. The accepted candidate is exactly representable as `int8_q6`.

```text
-1.00  -> q=-64
-0.75  -> q=-48
-0.50  -> q=-32
+0.25  -> q=+16
+0.75  -> q=+48
+1.00  -> q=+64
```

The verifier runs the reciprocal path with integer accumulation:

```text
bits -> hidden_acc -> code_acc -> mirror_hidden_acc -> logits_acc
```

Sign decode is invariant under the fixed positive scale, so the integer artifact
must exactly reproduce the float-search candidate.

## Current Deployment Meaning

`A-StableCopy16` remains the shipped/default AB codec A-block until compatibility
and integration checks explicitly promote this hidden-natural artifact.

This artifact proves that the hidden-natural candidate is not merely a float
playground object. It now has a compact integer representation suitable for the
next native integration pass.
