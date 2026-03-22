Legacy output-band probes archived during the open-beta canonicalization pass.

These files assumed a pre-projection graph surface with:
- `net.N`
- `net.out_start`
- explicit input/output neuron bands
- or old dense/scalar GPU compat fields

They are kept for historical reference only. The active `v4.2` line now uses:
- hidden-only graphs (`H`)
- fixed `input_projection` / `output_projection`
- projection-based readout
- `mutate()` as the only active mutation entrypoint
