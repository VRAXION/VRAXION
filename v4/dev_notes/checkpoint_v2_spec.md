# Checkpoint V2 Spec

## Purpose
- Deterministic resume with full state restoration.
- Architecture-safe eval/resume by storing explicit model build spec.
- Fail-fast on schema/version mismatch.

## Versioning
- `ckpt_version: 2`
- Loader is strict: non-`2` checkpoints are rejected.

## Required top-level keys
- `ckpt_version`
- `run_id`
- `step`
- `best_loss`
- `timestamp_utc`
- `model`
- `optimizer`
- `train_config_resolved`
- `model_config_resolved`
- `data_state`
- `sequence_state`
- `rng_state`
- `env`

## Model payload
- `model.type`: `instnct | transformer`
- `model.module`
- `model.class_name`
- `model.build_spec`: fully resolved init kwargs
- `model.state_dict`

## Optimizer payload
- `optimizer.class_name`
- `optimizer.state_dict`

## Data payload
- `data_state.data_dir`
- `data_state.seq_len`
- `data_state.batch_size`
- `data_state.embed_mode`
- `data_state.sequential`
- `data_state.file_manifest[]` with:
  - `path`
  - `size`
  - `mtime_ns`
  - `sha256_head_1mb`
- optional: `data_state.seq_offsets` (sequential mode)

## Sequence payload
- optional tensors/values:
  - `ring_state`
  - `ptr_state`
  - `hidden_state`
  - `bb_buf`
  - `bb_keys`
  - `bb_write_ptr`
  - `bb_steps`

## RNG payload
- `python_random_state`
- `numpy_random_state`
- `torch_cpu_rng_state`
- optional: `torch_cuda_rng_state_all`
- `cudnn_benchmark`
- `cudnn_deterministic`
- `torch_deterministic_algorithms`
- `dataset_rng_state`
- `eval_seed`

## Environment payload
- `python`
- `torch`
- `cuda`
- `hostname`
- `platform`
- `git_commit`

## Resume safety rules
- Deterministic mode (`training.deterministic_resume=true`) disables hot reload.
- Resume model is always built from `checkpoint.model.build_spec`, not current YAML.
- Data manifest drift emits warning or hard fail (`manifest_strict=true`).

## Tools
- Inspect checkpoint:
  - `python tools/inspect_checkpoint.py --ckpt training_output/ckpt_latest.pt`
  - `python tools/inspect_checkpoint.py --ckpt A.pt --against B.pt --print-build-spec`
