# E136 Assistant Text Seed Pack Download Report

```text
pack_id = e136_assistant_text_seed_pack
status  = local_download_and_normalization_complete
scope   = assistant/text seed data only; not model training or evidence pass
```

## Local Artifacts

```text
raw_root = target/datasets/e136_assistant_text_seed_pack/raw
normalized = target/datasets/e136_assistant_text_seed_pack/normalized/e136_assistant_text_skill_seed.jsonl
download_manifest = target/datasets/e136_assistant_text_seed_pack/download_manifest.json
normalized_manifest = target/datasets/e136_assistant_text_seed_pack/normalized_manifest.json

raw_size = 2.726 GiB
normalized_size = 2.430 GiB
normalized_rows = 447,766
normalized_sha256 = 7517fc747466228279137f8ab5f312475ded0216e4c9822ab97811952b95e5dc
sha256_first_256_rows = b200908430fabb01e46a1f13b10f5d64493ee289e623d16e84939274e49dca21
```

`target/` is gitignored, so raw and normalized datasets are local artifacts and
are not committed to the repository.

## Sources

```text
HuggingFaceH4/ultrachat_200k = 220,000 rows, MIT
Open-Orca/SlimOrca           = 120,000 rows, MIT
OpenAssistant/oasst2         = 37,766 rows, Apache-2.0
Anthropic/hh-rlhf            = 60,000 rows, MIT
HuggingFaceH4/no_robots      = 10,000 rows, CC-BY-NC-4.0
```

Source links:

- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- https://huggingface.co/datasets/Open-Orca/SlimOrca
- https://huggingface.co/datasets/OpenAssistant/oasst2
- https://huggingface.co/datasets/Anthropic/hh-rlhf
- https://huggingface.co/datasets/HuggingFaceH4/no_robots

## Normalized Mix

```text
train rows = 430,927
validation rows = 16,839

assistant_style = 447,766
instruction_following = 447,766
multi_turn_dialogue = 372,963
reasoning_instruction = 120,000
synthetic_multi_turn = 220,000
human_assistant_dialogue = 37,766
helpful_harmless = 60,000
preference_boundary = 60,000
refusal_or_boundary = 73,625
math_text_surface = 84,188
code_instruction = 83,423
summarization = 42,910
```

## Reproduction

```bash
python3 scripts/tools/prepare_e136_assistant_text_seed_pack.py
```

To reuse already-downloaded raw files and only regenerate normalized JSONL:

```bash
python3 scripts/tools/prepare_e136_assistant_text_seed_pack.py --skip-download
```

## Boundary

This pack is prepared for the next assistant/text operator farming and transfer
work. It does not prove open-domain assistant behavior, does not train neural
weights, does not promote any operator to Core/PermaCore/TrueGolden, and does
not change the current evidence anchor from E135.

`HuggingFaceH4/no_robots` is CC-BY-NC-4.0. Keep this pack in local
research/evidence workflows unless licensing is reviewed for a broader use.
