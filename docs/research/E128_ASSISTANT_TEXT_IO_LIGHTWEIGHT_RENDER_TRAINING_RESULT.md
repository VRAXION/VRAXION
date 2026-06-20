# E128 Assistant Text-IO Lightweight Render Training Result

```text
decision = e128_assistant_text_io_lightweight_render_training_confirmed
next = E129_ASSISTANT_PROMPT_GENERALIZATION_AND_LONGER_CONTEXT_SMOKE
boundary = deterministic corpus + action-policy/template renderer, not neural LLM/freeform generation
prompt_count = 320
train_action_accuracy = 1.000
validation_action_accuracy = 1.000
heldout_action_accuracy = 1.000
operator_trace_validity = 1.000
unsupported_answer_count = 0
wrong_refusal_count = 0
boundary_claim_violation_count = 0
```

## Summary

E128 builds the first no-download lightweight assistant-text training corpus on
top of the E127 governed operator library. The corpus is local and auditable:

```text
e127_smoke_seed = 40
e127_operator_derived = 88
repo_doc_grounded = 96
adversarial_boundary = 64
fineweb_derived_noise = 32
```

The split is deterministic:

```text
train = 160
validation = 64
heldout = 96
```

The run validates this bridge:

```text
prompt
-> scoped operator hints
-> action policy
-> evidence slots
-> guarded template response
```

This is a stronger text-IO smoke than the E127 8-prompt render sample because it
uses 320 prompts, explicit train/validation/heldout splits, adversarial boundary
cases, and local FineWeb-derived noise. It is still not neural LLM training,
not learned general model weights, and not freeform generation.

## Confirmed

```text
overall_action_accuracy = 1.000
operator_trace_validity = 1.000
unsupported_answer_rate = 0.000
wrong_refusal_rate = 0.000
boundary_claim_violation_rate = 0.000
```

The confirmed action classes are:

```text
ANSWER = 109
ASK_OR_DEFER = 50
ASK_PERMISSION_OR_SAFE_ALTERNATIVE = 5
DIAGNOSE_BOUNDARY = 18
NEXT_ACTION = 26
REFUSE_OR_BOUNDARY = 67
SUMMARIZE = 45
```

## Interpretation

E128 confirms that the current E127 operator-library artifacts can be converted
into a small assistant-style text-IO corpus without downloading a large external
chat dataset.

The useful claim is narrow:

```text
VRAXION can build and pass a governed, local, lightweight assistant-text
training/render smoke with deterministic action labels, operator traces,
evidence slots, and guarded template outputs.
```

The unsupported claims remain unchanged:

```text
open-domain chatbot = no
Gemma/GPT-level generation = no
production assistant = no
trained general weights = no
PermaCore / TrueGolden = no
consciousness / sentience = no
```

## Artifacts

```text
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
archived_public_artifact_sample_removed
```

## Reproduce

```powershell
python private_probe_runner_removed --out target/pilot_wave/e128_assistant_text_io_lightweight_render_training --sample-out archived_public_artifact_sample_removed
```
