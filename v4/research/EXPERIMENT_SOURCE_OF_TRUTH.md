# v4.1 Experiment Source of Truth

`origin/v4.1` is the active proving-ground branch for the no-loss / self-wiring line.
It is intentionally broader and less locked than `origin/nightly`.

Branch policy:

- `main`
  - frozen historical old-old architecture line
- `origin/nightly`
  - curated ring/loss trunk
  - only validated outcomes get promoted here
- `origin/v4.1`
  - experimental trunk
  - new self-wiring, capacitor, abstain/introspection, and related no-loss ideas live here first

Promotion rule:

- ideas proven in `v4.1` may later be promoted to `nightly`
- ideas that fail stay documented in `v4.1` or in archive tags
- raw branch-local clutter should be normalized into readable docs and small result artifacts before promotion

Current canonical experiment areas:

- `v4/research/v22/`
  - capacitor-neuron self-wiring results and supporting tests
- `v22_ternary/`
  - ternary-mask experiment line preserved as experiment evidence
- `v23_instnct_lm/`
  - CPU byte-level language-model experiment preserved as experiment evidence

Legacy policy:

- older lines like `v19b` and `v21` are not promoted into the active experiment trunk by default
- if they matter again, pull them from archive tags or move them under a clearly-marked legacy area instead of mixing them into current canonical experiment paths
