# ProtocolFramingIngressPocket v001

Frozen pocket contract:

- Reads anonymous binary stream bits.
- Uses START/LENGTH/CRC/END framing hygiene.
- Requires requested-feature compatibility before committing to Flow Field.
- Does not own world-specific feature-codebook mapping; that is an adapter.
- Must reject rather than write when protocol checks fail.
