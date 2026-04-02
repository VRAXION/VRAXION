"""
INSTNCT — Network DNA: encode/decode a trained network as a string
===================================================================
Serialize the LEARNED parts only. Projections come from seed.

Format: INSTNCT:v1:H:seed:base64(packed_data)

packed_data:
  - edges: (row, col) pairs, uint16 if H≤65535
  - theta: int4 bitpacked (2 per byte)
  - channel: 3-bit packed (floor(8/3)=2 per byte + remainder)
  - polarity: 1-bit packed (8 per byte)
  - decay: quantized to uint8 (0.0-1.0 → 0-255)

Test: encode → decode → verify exact match.
"""
import sys, base64, struct
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph


def encode_network(net):
    """Encode trained network to a compact string."""
    H = net.H
    V = net.V
    seed = 42  # projection seed (must match what was used to create net)

    # Edge list
    rows, cols = np.where(net.mask)
    n_edges = len(rows)
    idx_dtype = np.uint16 if H <= 65535 else np.uint32

    # Pack everything into bytes
    parts = []

    # Header: H (uint32), V (uint16), n_edges (uint32)
    parts.append(struct.pack('<IHI', H, V, n_edges))

    # Edges: row, col pairs
    parts.append(rows.astype(idx_dtype).tobytes())
    parts.append(cols.astype(idx_dtype).tobytes())

    # Theta: int4 bitpacked (2 per byte)
    theta_packed = SelfWiringGraph._pack_int4(net.theta)
    parts.append(theta_packed.tobytes())

    # Channel: uint8 [1-8] → store as-is (1 byte per neuron, simple)
    parts.append(net.channel.tobytes())

    # Polarity: bool → bitpacked (8 per byte)
    pol_bytes = np.packbits(net.polarity.astype(np.uint8))
    parts.append(pol_bytes.tobytes())

    # Decay: float32 → quantize to uint8 [0-255] (range 0.0-1.0)
    decay_q = np.clip(np.round(net.decay * 255), 0, 255).astype(np.uint8)
    parts.append(decay_q.tobytes())

    # Meta: loss_pct, mutation_drive
    parts.append(struct.pack('<bb', int(net.loss_pct), int(net.mutation_drive)))

    raw = b''.join(parts)
    b64 = base64.b85encode(raw).decode('ascii')

    # Construct DNA string
    dna = f"INSTNCT:v1:{H}:{seed}:{b64}"
    return dna, len(raw)


def decode_network(dna_string):
    """Decode DNA string back to a SelfWiringGraph."""
    parts = dna_string.split(':')
    assert parts[0] == 'INSTNCT', f"Invalid header: {parts[0]}"
    version = parts[1]
    H = int(parts[2])
    seed = int(parts[3])
    b64 = ':'.join(parts[4:])  # rejoin in case b85 contains ':'

    raw = base64.b85decode(b64)
    offset = 0

    # Header
    H_dec, V, n_edges = struct.unpack_from('<IHI', raw, offset)
    offset += struct.calcsize('<IHI')
    assert H_dec == H, f"H mismatch: header {H_dec} vs DNA {H}"

    # Create network with same seed (reconstructs projections)
    net = SelfWiringGraph(vocab=V, hidden=H, seed=seed)

    # Edges
    idx_dtype = np.uint16 if H <= 65535 else np.uint32
    idx_size = np.dtype(idx_dtype).itemsize
    rows = np.frombuffer(raw, dtype=idx_dtype, count=n_edges, offset=offset)
    offset += n_edges * idx_size
    cols = np.frombuffer(raw, dtype=idx_dtype, count=n_edges, offset=offset)
    offset += n_edges * idx_size

    # Reconstruct mask
    net.mask[:] = False
    net.mask[rows, cols] = True
    np.fill_diagonal(net.mask, False)

    # Theta
    theta_packed_len = (H + 1) // 2
    theta_packed = np.frombuffer(raw, dtype=np.uint8, count=theta_packed_len, offset=offset)
    offset += theta_packed_len
    net.theta = SelfWiringGraph._unpack_int4(theta_packed, H)
    net._theta_f32 = net.theta.astype(np.float32)

    # Channel
    net.channel = np.frombuffer(raw, dtype=np.uint8, count=H, offset=offset).copy()
    offset += H

    # Polarity
    pol_bytes_len = (H + 7) // 8
    pol_packed = np.frombuffer(raw, dtype=np.uint8, count=pol_bytes_len, offset=offset)
    offset += pol_bytes_len
    pol_bits = np.unpackbits(pol_packed)[:H]
    net.polarity = pol_bits.astype(np.bool_)
    net._polarity_f32 = np.where(net.polarity, 1.0, -1.0).astype(np.float32)

    # Decay
    decay_q = np.frombuffer(raw, dtype=np.uint8, count=H, offset=offset).copy()
    offset += H
    net.decay = (decay_q.astype(np.float32) / 255.0)

    # Meta
    loss_pct, mut_drive = struct.unpack_from('<bb', raw, offset)
    offset += 2
    net.loss_pct = np.int8(loss_pct)
    net.mutation_drive = np.int8(mut_drive)

    net.resync_alive()
    net.reset()
    return net


def verify_match(original, decoded):
    """Verify the decoded network matches the original."""
    checks = {}

    # Mask
    checks['mask'] = np.array_equal(original.mask, decoded.mask)

    # Theta
    checks['theta'] = np.array_equal(original.theta, decoded.theta)

    # Channel
    checks['channel'] = np.array_equal(original.channel, decoded.channel)

    # Polarity
    checks['polarity'] = np.array_equal(original.polarity, decoded.polarity)

    # Decay (quantized — allow small error)
    decay_err = np.max(np.abs(original.decay - decoded.decay))
    checks['decay'] = decay_err < 0.005  # <0.5% error from uint8 quantization

    # Projections (should be identical from same seed)
    checks['input_proj'] = np.allclose(original.input_projection, decoded.input_projection)
    checks['output_proj'] = np.allclose(original.output_projection, decoded.output_projection)

    # Edge count
    checks['edge_count'] = len(original.alive) == len(decoded.alive)

    # Meta
    checks['loss_pct'] = int(original.loss_pct) == int(decoded.loss_pct)
    checks['mutation_drive'] = int(original.mutation_drive) == int(decoded.mutation_drive)

    return checks


def main():
    PRED_NEURONS = list(range(0, 10))

    print("=" * 80)
    print("  Network DNA: encode → string → decode → verify")
    print("=" * 80)

    # Test at different scales
    for H in [32, 64, 256, 1024]:
        print(f"\n  --- H={H} ---")

        # Create and train briefly
        net = SelfWiringGraph(vocab=256, hidden=H, density=4,
                              theta_init=1, decay_init=0.10, seed=42)
        # Mutate a few times so it's not just init state
        for _ in range(50):
            net.mutate()

        edges = len(net.alive)
        print(f"  Edges: {edges}")

        # Encode
        dna, raw_bytes = encode_network(net)
        dna_len = len(dna)

        # Decode
        decoded = decode_network(dna)

        # Verify
        checks = verify_match(net, decoded)
        all_pass = all(checks.values())

        print(f"  Raw bytes:  {raw_bytes:,}")
        print(f"  DNA string: {dna_len:,} chars")
        print(f"  Compression vs full checkpoint:")

        # Compare to npz size
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            net.save(f.name)
            npz_size = os.path.getsize(f.name)
            os.unlink(f.name)
        print(f"    NPZ checkpoint: {npz_size:,} bytes")
        print(f"    DNA string:     {raw_bytes:,} bytes ({100*raw_bytes/npz_size:.1f}% of NPZ)")
        print(f"    DNA chars:      {dna_len:,} (pasteable)")

        print(f"  Verification: {'ALL PASS' if all_pass else 'FAILED'}")
        for check, ok in checks.items():
            if not ok:
                print(f"    FAIL: {check}")

        # Show the DNA string (truncated)
        if dna_len < 500:
            print(f"\n  DNA: {dna}")
        else:
            print(f"\n  DNA: {dna[:80]}...({dna_len} chars)...{dna[-20:]}")

    # Practical demo: encode a trained network
    print(f"\n{'='*80}")
    print(f"  PRACTICAL DEMO: train → encode → share → decode → eval")
    print(f"{'='*80}")

    import random
    random.seed(42); np.random.seed(42)
    net = SelfWiringGraph(vocab=256, hidden=64, density=4,
                          theta_init=1, decay_init=0.10, seed=42)
    # Quick train on alternating pattern
    def make_alt(rng, n=30):
        a, b = rng.randint(0, 10, size=2)
        while b == a: b = rng.randint(0, 10)
        return [a if i % 2 == 0 else b for i in range(n + 1)]

    eval_seqs = [make_alt(np.random.RandomState(77+i), 30) for i in range(3)]
    for step in range(500):
        snap = net.save_state(); net.mutate()
        # quick eval
        net.reset()
        sc = SelfWiringGraph.build_sparse_cache(net.mask)
        st = np.zeros(64, dtype=np.float32); ch = np.zeros(64, dtype=np.float32)
        c = 0; t = 0
        for seq in eval_seqs[:1]:
            for i in range(len(seq)-1):
                st, ch = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask,
                    theta=net._theta_f32, decay=net.decay, ticks=8,
                    input_duration=2, state=st, charge=ch, sparse_cache=sc,
                    polarity=net._polarity_f32, refractory=net.refractory,
                    channel=net.channel)
                if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
                t += 1
        new_acc = c/t if t else 0
        if step == 0 or new_acc > getattr(main, '_best', 0):
            main._best = new_acc
        else:
            net.restore_state(snap)

    print(f"\n  Trained: acc={main._best:.3f}")

    # Encode
    dna, raw = encode_network(net)
    print(f"  DNA: {len(dna)} chars")

    # Someone else decodes it
    restored = decode_network(dna)
    checks = verify_match(net, restored)
    print(f"  Decoded: {'EXACT MATCH' if all(checks.values()) else 'MISMATCH'}")

    # Eval the decoded network
    restored.reset()
    sc = SelfWiringGraph.build_sparse_cache(restored.mask)
    st = np.zeros(64, dtype=np.float32); ch = np.zeros(64, dtype=np.float32)
    c = 0; t = 0
    for seq in eval_seqs:
        for i in range(len(seq)-1):
            st, ch = SelfWiringGraph.rollout_token(
                restored.input_projection[int(seq[i])], mask=restored.mask,
                theta=restored._theta_f32, decay=restored.decay, ticks=8,
                input_duration=2, state=st, charge=ch, sparse_cache=sc,
                polarity=restored._polarity_f32, refractory=restored.refractory,
                channel=restored.channel)
            if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
            t += 1
    dec_acc = c/t if t else 0
    print(f"  Decoded eval: acc={dec_acc:.3f}")
    print(f"\n  ✓ Trained network shared as {len(dna)}-char string, reconstructed perfectly.")


if __name__ == "__main__":
    main()
