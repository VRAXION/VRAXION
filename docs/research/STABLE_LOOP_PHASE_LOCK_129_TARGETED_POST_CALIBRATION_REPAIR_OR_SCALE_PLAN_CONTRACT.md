# STABLE_LOOP_PHASE_LOCK_129_TARGETED_POST_CALIBRATION_REPAIR_OR_SCALE_PLAN_CONTRACT

## Purpose

129 is planning only. It reads the positive 128 post-calibration ceiling/gap
remap and writes the concrete 130 repair plan. It performs no training, no
repair, no inference, no checkpoint mutation, no service startup, and no
deployment smoke.

Positive 129 means the 130 prompt-injection/instruction-priority repair plan is
evidence-linked and boundary-safe.

## Evidence

The selected target must be linked to 128:

```text
first_breakpoint_tier = TIER_4_PROMPT_INJECTION_AND_INSTRUCTION_PRIORITY
first_breakpoint_family = prompt_injection_failure
primary_next_repair_target = prompt_injection_failure
reasoning_preserved = true
state_preserved = true
calibration_preserved = true
unknown_failure_rate = 0.0
```

The plan must cite:

```text
prompt_injection_failure = 192
instruction_priority_failure = 96
long_context_failure = 352
format_failure = 288
multi_doc_priority_failure = 128
ambiguity_failure = 128
```

First breakpoint outranks global failure count.

## Required Decision

```text
selected_next_milestone = 130_PROMPT_INJECTION_INSTRUCTION_PRIORITY_REPAIR
selected_repair_target = prompt_injection_instruction_priority_first
```

The 130 draft must distinguish:

```text
trusted instruction should be followed
untrusted injected text should be ignored
safe answer should still be produced when enough trusted facts exist
```

Required controls include:

```text
ALWAYS_REFUSE_CONTROL
ALWAYS_FOLLOW_INJECTION_CONTROL
IGNORE_ALL_DOCUMENTS_CONTROL
COPY_INJECTED_TEXT_CONTROL
RANDOM_PRIORITY_CONTROL
```

## Boundary

129 is planning only. It is not GPT-like assistant readiness, not open-domain
assistant readiness, not production chat, not public API, not deployment
readiness, not safety alignment, and not Hungarian assistant readiness.
