"""
LCX Forensics: Load milestone checkpoints on CPU, run eval, extract scratchpad state.
Traces the formation of the R2 dark stripe and correlates with accuracy changes.
"""
import sys, os, random, json
import torch
import torch.nn.functional as F

sys.path.insert(0, r"S:\AI\work\VRAXION_DEV\Diamond Code")
from swarm_model import SwarmByteRingModel

CKPT_DIR = r"S:\AI\work\VRAXION_DEV\Diamond Code\checkpoints\swarm"

# Milestone steps to evaluate (spanning R2's descent phases)
MILESTONES = [99, 299, 499, 1000, 1500, 1999, 2500]

# Fixed eval seed for reproducibility across checkpoints
EVAL_SEED = 777
N_EVAL = 500
N_LCX_WARMUP = 20  # forward passes to let LCX settle before reading


def int_to_bits(x, num_bits=8):
    return torch.tensor([(x >> i) & 1 for i in range(num_bits)], dtype=torch.float32)


def generate_eval_batch(n_samples, num_bits=8, seq_len=16, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    max_val = 2 ** num_bits - 1
    op_codes = {
        'add': int_to_bits(1 << 0, num_bits),
        'and': int_to_bits(1 << 1, num_bits),
        'or':  int_to_bits(1 << 2, num_bits),
        'xor': int_to_bits(1 << 3, num_bits),
    }
    op_names = ['add', 'and', 'or', 'xor']
    x_batch, y_batch, op_indices = [], [], []
    for _ in range(n_samples):
        op_name = random.choice(op_names)
        op_idx = op_names.index(op_name)
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        if op_name == 'add':    result = (a + b) % (2 ** num_bits)
        elif op_name == 'and':  result = a & b
        elif op_name == 'or':   result = a | b
        else:                   result = a ^ b
        x_seq = torch.zeros(seq_len, num_bits)
        x_seq[0] = int_to_bits(a, num_bits)
        x_seq[1] = int_to_bits(b, num_bits)
        x_seq[2] = op_codes[op_name]
        y_seq = torch.zeros(seq_len, num_bits)
        y_seq[0] = int_to_bits(a, num_bits)
        y_seq[1] = int_to_bits(b, num_bits)
        y_seq[2] = int_to_bits(result, num_bits)
        x_batch.append(x_seq)
        y_batch.append(y_seq)
        op_indices.append(op_idx)
    return torch.stack(x_batch), torch.stack(y_batch), torch.tensor(op_indices)


def build_model(cfg):
    memory_size = cfg.get('memory_size', cfg['embedding_dim'])
    model = SwarmByteRingModel(
        num_memory_positions=memory_size,
        embedding_dim=cfg['embedding_dim'],
        num_beings=cfg['num_beings'],
        depth=cfg['depth'],
        num_bits=cfg['num_bits'],
        combiner_mode=cfg.get('combiner', 'masked'),
        bits_per_being=cfg.get('bits_per_being', 8),
        min_coverage=cfg.get('min_coverage', 1),
        mask_seed=cfg.get('mask_seed', 42),
        fibonacci=cfg.get('fibonacci', True),
        combinatorial=cfg.get('combinatorial', False),
        think_ticks=0,
        temporal_fibonacci=cfg.get('temporal_fibonacci', False),
        capacity_fibonacci=cfg.get('capacity_fibonacci', False),
        max_hidden=cfg.get('max_hidden', 4096),
        min_hidden=cfg.get('min_hidden', 128),
        full_view=cfg.get('full_view', False),
        use_lcx=cfg.get('use_lcx', False),
    )
    return model


def eval_checkpoint(ckpt_path, x_eval, y_eval, op_indices):
    """Load checkpoint, run eval, return metrics + LCX state."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    cfg = ckpt['config']
    step = ckpt['step']

    model = build_model(cfg)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    num_bits = cfg['num_bits']
    has_lcx = cfg.get('use_lcx', False) and model.lcx is not None

    # Warm up LCX with multiple forward passes on small batches
    if has_lcx:
        with torch.no_grad():
            for i in range(N_LCX_WARMUP):
                chunk = x_eval[i * 25:(i + 1) * 25]
                if len(chunk) == 0:
                    break
                model(chunk)

    # Main eval pass
    with torch.no_grad():
        try:
            output, stats = model(x_eval, return_stats=True, return_being_outputs=True)
        except Exception:
            output = model(x_eval)
            stats = {}

    # Bit accuracy and byte match at result position (pos 2)
    eval_pos = 2
    pred_bits = (output[:, eval_pos, :] > 0.0).float()
    target_bits = y_eval[:, eval_pos, :].float()
    byte_match = (pred_bits == target_bits).all(dim=-1).float().mean().item()
    bit_acc = (pred_bits == target_bits).float().mean().item()

    # Per-op accuracy
    op_names = ['add', 'and', 'or', 'xor']
    per_op = {}
    for oi, oname in enumerate(op_names):
        mask = op_indices == oi
        if mask.sum() > 0:
            per_op[oname] = (pred_bits[mask] == target_bits[mask]).float().mean().item()

    # LCX state
    lcx_grid = None
    lcx_row_means = None
    if has_lcx:
        lcx_state = model.lcx.clone()
        lcx_grid = lcx_state.view(num_bits, num_bits)
        lcx_row_means = lcx_grid.mean(dim=1).tolist()

    # Per-being accuracy
    being_accs = []
    if 'being_outputs' in stats:
        bo = stats['being_outputs']  # [num_beings, T, B, num_bits]
        for i in range(cfg['num_beings']):
            bp = (bo[i][eval_pos, :, :] > 0.0).float()
            ba = (bp == target_bits).float().mean().item()
            being_accs.append(ba)

    return {
        'step': step,
        'bit_acc': bit_acc,
        'byte_match': byte_match,
        'per_op': per_op,
        'lcx_grid': lcx_grid,
        'lcx_row_means': lcx_row_means,
        'being_accs': being_accs,
        'num_beings': cfg['num_beings'],
        'use_lcx': has_lcx,
        'config': cfg,
    }


def format_lcx_grid(grid, num_bits=8):
    """Pretty-print the 8x8 LCX grid."""
    lines = []
    header = "     " + "  ".join(f"C{c}" for c in range(num_bits))
    lines.append(header)
    for r in range(num_bits):
        vals = " ".join(f"{grid[r, c]:+.3f}" for c in range(num_bits))
        lines.append(f"R{r}  {vals}")
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  LCX FORENSICS — Scratchpad Evolution Across Checkpoints")
    print("=" * 72)

    # Check which milestones exist
    available = []
    for step in MILESTONES:
        path = os.path.join(CKPT_DIR, f"checkpoint_step_{step}.pt")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            available.append((step, path, size_mb))
            print(f"  [OK] step {step:>6d}  ({size_mb:.1f} MB)")
        else:
            print(f"  [--] step {step:>6d}  NOT FOUND")

    if not available:
        print("\nNo checkpoints found!")
        return

    # Check config consistency — identify which belong to the same run
    print(f"\n{'='*72}")
    print("  CONFIG CHECK — Identifying current run checkpoints")
    print(f"{'='*72}")
    configs = {}
    for step, path, size_mb in available:
        ckpt = torch.load(path, weights_only=False, map_location='cpu')
        cfg = ckpt['config']
        key = (cfg['num_beings'], cfg['embedding_dim'], cfg['depth'],
               cfg.get('combiner', '?'), cfg.get('use_lcx', False))
        configs[step] = key
        print(f"  step {step:>6d}: {cfg['num_beings']}b {cfg['embedding_dim']}d "
              f"depth={cfg['depth']} {cfg.get('combiner','?')} "
              f"lcx={cfg.get('use_lcx', False)} fibo={cfg.get('capacity_fibonacci', False)} "
              f"full_view={cfg.get('full_view', False)}")
        del ckpt

    # Group by config
    config_groups = {}
    for step, key in configs.items():
        config_groups.setdefault(key, []).append(step)
    print(f"\n  {len(config_groups)} distinct config(s) found:")
    for key, steps in config_groups.items():
        print(f"    {key} -> steps {steps}")

    # Generate fixed eval data (use config from first available checkpoint)
    first_ckpt = torch.load(available[0][1], weights_only=False, map_location='cpu')
    first_cfg = first_ckpt['config']
    del first_ckpt

    num_bits = first_cfg['num_bits']
    seq_len = first_cfg.get('seq_len', 16)
    print(f"\n  Generating eval data: {N_EVAL} samples, {num_bits} bits, seq_len={seq_len}")
    x_eval, y_eval, op_indices = generate_eval_batch(N_EVAL, num_bits, seq_len, EVAL_SEED)

    # Evaluate each checkpoint
    results = []
    for step, path, size_mb in available:
        print(f"\n{'='*72}")
        print(f"  EVALUATING STEP {step}")
        print(f"{'='*72}")

        try:
            r = eval_checkpoint(path, x_eval, y_eval, op_indices)
            results.append(r)

            print(f"  Bit accuracy:  {r['bit_acc']*100:6.2f}%")
            print(f"  Byte match:    {r['byte_match']*100:6.2f}%")
            print(f"  Per-op: ", end="")
            for op, acc in r['per_op'].items():
                print(f"{op}={acc*100:.1f}% ", end="")
            print()

            if r['being_accs']:
                print(f"  Being accs: ", end="")
                for i, ba in enumerate(r['being_accs']):
                    print(f"B{i}={ba*100:.1f}% ", end="")
                print()

            if r['lcx_grid'] is not None:
                print(f"\n  LCX Scratchpad (after {N_LCX_WARMUP} warmup passes):")
                print(f"  {format_lcx_grid(r['lcx_grid'], num_bits)}")
                print(f"\n  Row means: ", end="")
                for i, rm in enumerate(r['lcx_row_means']):
                    marker = " <<<" if abs(rm) > 0.3 else ""
                    print(f"R{i}={rm:+.3f}{marker} ", end="")
                print()
            else:
                print(f"  LCX: not active (use_lcx={r['use_lcx']})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    if results:
        print(f"\n{'='*72}")
        print("  EVOLUTION SUMMARY")
        print(f"{'='*72}")
        print(f"  {'Step':>6s}  {'Bit%':>6s}  {'Byte%':>6s}  {'R0':>7s}  {'R1':>7s}  {'R2':>7s}  {'R3':>7s}  {'R4':>7s}  {'R5':>7s}  {'R6':>7s}  {'R7':>7s}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*6}  " + "  ".join(['-'*7]*8))
        for r in results:
            row = f"  {r['step']:>6d}  {r['bit_acc']*100:>5.1f}%  {r['byte_match']*100:>5.1f}%"
            if r['lcx_row_means']:
                for rm in r['lcx_row_means']:
                    row += f"  {rm:>+7.4f}"
            else:
                row += "  (no LCX)" + " " * 55
            print(row)

        # R2 specific analysis
        r2_values = [(r['step'], r['lcx_row_means'][2]) for r in results if r['lcx_row_means']]
        if r2_values:
            print(f"\n  R2 TRAJECTORY:")
            for step, val in r2_values:
                bar_len = int(abs(val) * 40)
                bar = "#" * bar_len
                direction = "<" if val < 0 else ">"
                print(f"    step {step:>6d}: {val:>+.4f} {direction}{bar}")


if __name__ == "__main__":
    main()
