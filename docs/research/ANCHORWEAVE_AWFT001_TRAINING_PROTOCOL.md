# AnchorWeave AWFT-001 Training Protocol

## Purpose

AWFT-001 training tests whether AnchorWeave input structure creates a measurable
same-base-model learning effect on navigation / route-memory disambiguation.

The experiment compares one trainable HuggingFace model across zero-shot and
four LoRA/SFT arms. Ollama zero-shot remains useful historical context, but it
is not the training baseline because Ollama is not the fine-tuning backend.

This is a local format-transfer test. It is not evidence that AnchorWeave is
useful at scale, and it is not a consciousness claim.

## Arms

All trained arms use the same base model, seed, train row count, LoRA config,
max sequence length, and evaluation prompts.

- `zero_shot`: same HuggingFace base model, no adapter
- `plain_qa`: compact Q&A-style input, structured JSON target
- `rich_prose`: prose input, same structured JSON target
- `anchorweave_sft`: AnchorWeave-style structured input, same structured JSON target
- `shuffled_anchorweave`: AnchorWeave-style input with intentionally bad target

The primary comparison uses common `eval_prompts.jsonl` for every arm. This is
a deliberate transfer stress test. A negative AnchorWeave result can mean the
format failed to transfer into the common prompt, not only that structured input
has no useful training signal. The runner provides an optional view-matched
diagnostic for this caveat.

## Training Controls

Completion-only loss is required. Prompt tokens are masked with `labels=-100`,
and loss is computed only on the assistant JSON completion tokens.

Training examples use this chat shape:

```text
system: Return only valid JSON. No markdown. No explanation.
user: <training view prompt>
assistant: <compact structured JSON target>
```

The runner reports per arm:

- average prompt tokens
- average completion tokens
- max sequence length
- prompt truncation count
- completion truncation count
- LoRA target module strategy and target module count

Completion truncation is a hard error. Prompt truncation is counted and warned.
The implementation truncates prompt tokens only when necessary; completion
tokens are never truncated.

LoRA target modules are selected robustly. The preferred PEFT strategy is
`target_modules="all-linear"`. If that fails, the trainer dynamically detects
Linear leaf module names and hard fails if no target modules exist.

## Default Run

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Fallback if unavailable:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

Default hyperparameters:

```json
{
  "seed": 2026,
  "epochs": 8,
  "learning_rate": 0.0002,
  "batch_size": 1,
  "grad_accum": 8,
  "max_seq_len": 2048,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

Dry-run:

```bash
python tools/anchorweave/run_awft001_ab.py --out target/anchorweave/awft001_runs/dryrun --seed 2026 --model Qwen/Qwen2.5-0.5B-Instruct --dry-run --max-train-rows 4 --max-eval-rows 4 --epochs 1
```

Full run:

```bash
python tools/anchorweave/run_awft001_ab.py --out target/anchorweave/awft001_runs/first_ab --seed 2026 --model Qwen/Qwen2.5-0.5B-Instruct
```

The Python runner does not run `git checkout`, `git pull`, or any other git
operation. It assumes the repository is already in the intended state.

## Outputs

Generated outputs stay under:

```text
target/anchorweave/awft001_runs/
```

The runner writes:

- zero-shot predictions, invalid rows, and metrics
- four LoRA adapters
- per-arm predictions, invalid rows, and metrics
- `scoreboard.json`
- `scoreboard.md`
- dependency and CUDA metadata

Do not commit adapters, predictions, scoreboards, or generated target files.
Do not write under `data/anchorweave/cells/`.

## Interpretation

Strong positive:

```text
anchorweave_sft > rich_prose
anchorweave_sft > plain_qa
anchorweave_sft >> shuffled_anchorweave
```

The important metrics are:

- `symbol_reject_f1`
- `counterfactual_accuracy`
- `salience_high_f1`
- `salience_low_f1`
- low or unchanged `overcommitment_rate`

Weak positive:

```text
anchorweave_sft beats plain_qa but ties rich_prose
```

This suggests prose may already carry much of the useful signal.

Negative:

```text
anchorweave_sft does not beat rich_prose or plain_qa
```

This suggests the current AnchorWeave format may be too verbose, poorly aligned,
or not adding useful signal under the common-eval stress test.

Failed control:

```text
shuffled_anchorweave approximately matches anchorweave_sft
```

This suggests the experiment is broken or the model learned style/format rather
than correct grounding content.

Overcommitment failure:

```text
anchorweave_sft improves action accuracy but increases overcommitment_rate
```

This is not acceptable. It learned confidence, not grounded disambiguation.
