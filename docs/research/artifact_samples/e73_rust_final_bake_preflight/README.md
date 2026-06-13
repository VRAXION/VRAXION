# E73 Rust Final Bake Preflight Samples

These files mirror the generated artifact shape from:

```text
target/pilot_wave/e73_rust_final_bake_preflight/
```

Sample files:

```text
final_bake_results_sample.json
progress_sample.jsonl
report_sample.md
```

Expected sample result:

```text
passed = true
body_passed = body_cases
text_passed = text_cases
registry_passed = registry_cases
manager_mutation_passed = manager_mutation_cases
library_passed = library_cases
resume_passed = true
final_checksum_match = true
bad_commit_rate = 0
unsafe_promotion_rate = 0
```

This sample pack is for artifact-shape validation and documentation. The
generated target directory remains the evidence source for full local runs.
