"""
Quick eval: load latest checkpoint, test against all traindat files.
Runs on CPU to avoid interfering with active GPU training.
"""
import sys, os, time, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from swarm_model import SwarmByteRingModel
from traindat_loader import load_batch_from_file

STEP_TIMEOUT = 60  # seconds per dataset eval

# All datasets to evaluate
DATASETS = [
    ('constant256',    'Constant',    'data/traindat/', 'data/golden/'),
    ('copy_echo256',   'Copy Echo',   'data/traindat/', 'data/golden/'),
    ('echo256',        'Echo',        'data/traindat/', None),
    ('not256',         'Bitwise NOT', 'data/traindat/', None),
    ('count256',       'Counter',     'data/traindat/', None),
    ('shift256',       'Bit Shift',   'data/traindat/', None),
    ('denoise256',     'Denoise',     'data/traindat/', None),
    ('fib256',         'Fibonacci',   'data/traindat/', None),
    ('delay_echo256',  'Delay Echo',  'data/traindat/', None),
]

def find_traindat(base_dir, key, primary_dir, fallback_dir):
    fname = key + '.traindat'
    p = os.path.join(base_dir, primary_dir, fname)
    if os.path.exists(p):
        return p
    if fallback_dir:
        p2 = os.path.join(base_dir, fallback_dir, fname)
        if os.path.exists(p2):
            return p2
    return None

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ckpt_dir = os.path.join(base, 'checkpoints', 'curriculum_v2')

    # Find best checkpoint (golden first, then latest draft)
    golden_dir = os.path.join(ckpt_dir, 'golden')
    drafts_dir = os.path.join(ckpt_dir, 'drafts')

    best_ckpt = None
    best_step = -1

    for d in [golden_dir, drafts_dir]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.pt'):
                # Extract step from filename
                import re
                m = re.search(r'step_?(\d+)', f)
                if m:
                    s = int(m.group(1))
                    if s > best_step:
                        best_step = s
                        best_ckpt = os.path.join(d, f)

    if not best_ckpt:
        print('ERROR: No checkpoint found')
        sys.exit(1)

    print(f'Loading checkpoint: {os.path.basename(best_ckpt)} (step {best_step})')
    print(f'Running on CPU (training still active on GPU)')
    print()

    # Load checkpoint to get config
    t0 = time.time()
    checkpoint = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})
    print(f'Checkpoint loaded in {time.time()-t0:.1f}s')

    # Create model from config
    # lcx_level_slots can be int or comma-sep string in config
    raw_level_slots = cfg.get('lcx_level_slots', '2000')
    if isinstance(raw_level_slots, str):
        level_slots = [int(x) for x in raw_level_slots.split(',')]
    elif isinstance(raw_level_slots, (int, float)):
        level_slots = [int(raw_level_slots)]
    else:
        level_slots = list(raw_level_slots)

    model = SwarmByteRingModel(
        num_memory_positions=cfg.get('memory_size', 62),
        embedding_dim=cfg.get('embedding_dim', 6180),
        num_beings=cfg.get('num_beings', 1),
        depth=cfg.get('depth', 6),
        num_bits=cfg.get('num_bits', 8),
        think_ticks=1,
        use_lcx=cfg.get('use_lcx', True),
        lcx_mode=cfg.get('lcx_mode', 'hash'),
        lcx_num_slots=level_slots[0],
        lcx_key_dim=cfg.get('lcx_key_dim', 618),
        lcx_top_k=cfg.get('lcx_top_k', 2),
        lcx_num_levels=cfg.get('lcx_num_levels', 1),
        lcx_level_slots=level_slots,
        attention_radius=cfg.get('attention_radius', 6),
        num_pointers=cfg.get('num_pointers', 1),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f'Model created: {sum(p.numel() for p in model.parameters()):,} params')
    print()

    # Eval params
    batch_size = 10
    seq_len = cfg.get('seq_len', 62)
    num_bits = cfg.get('num_bits', 8)
    n_eval = 5  # average over 5 batches for stability

    print(f'{"Dataset":<16} {"bit_acc":>8} {"byte_match":>10} {"loss":>8}  {"status"}')
    print('-' * 62)

    results = []
    for key, name, primary, fallback in DATASETS:
        fpath = find_traindat(base, key, primary, fallback)
        if not fpath:
            print(f'{name:<16} {"--":>8} {"--":>10} {"--":>8}  file not found')
            continue

        t0 = time.time()
        print(f'  evaluating {name}...', end='', flush=True)

        bit_accs = []
        byte_matches = []
        losses = []

        try:
            with torch.no_grad():
                for ei in range(n_eval):
                    x, y, mask = load_batch_from_file(
                        fpath,
                        n_samples=batch_size,
                        seq_len=seq_len,
                        num_bits=num_bits,
                        seed=9999 + ei * 137,
                        binary_bits_mode=True,
                    )
                    # Mask padding
                    x = x.where(mask.bool(), torch.tensor(-1.0))

                    out, stats = model(x, return_stats=True, return_being_outputs=True)

                    # Compute loss
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        (out * mask).reshape(-1),
                        (y * mask).reshape(-1),
                        reduction='mean'
                    ).item()

                    # Compute bit accuracy at eval position
                    eval_pos = min(2, out.size(1) - 1)
                    pred_bits = (torch.sigmoid(out) > 0.5).float()
                    bit_correct = (pred_bits == y).float()

                    # Masked bit accuracy (only real bits)
                    data_mask = mask[:, eval_pos, :].bool()
                    if data_mask.any():
                        ba = bit_correct[:, eval_pos, :][data_mask].float().mean().item()
                    else:
                        ba = bit_correct[:, eval_pos, :].mean().item()
                    bit_accs.append(ba)

                    # Byte match (group 8 bits)
                    bc = bit_correct[:, eval_pos, :]
                    if num_bits >= 8:
                        bm = bc.reshape(bc.size(0), -1, 8).all(dim=-1).float().mean().item()
                    else:
                        bm = bc.all(dim=-1).float().mean().item()
                    byte_matches.append(bm)
                    losses.append(loss)

                    elapsed = time.time() - t0
                    if elapsed > STEP_TIMEOUT:
                        print(f'\r{name:<16} TIMEOUT after {elapsed:.0f}s')
                        break

            avg_ba = sum(bit_accs) / len(bit_accs)
            avg_bm = sum(byte_matches) / len(byte_matches)
            avg_loss = sum(losses) / len(losses)
            elapsed = time.time() - t0

            # Status label
            if avg_ba > 0.95:
                status = 'MASTERED'
            elif avg_ba > 0.70:
                status = 'learning'
            elif avg_ba > 0.55:
                status = 'signal'
            else:
                status = 'baseline'

            print(f'\r{name:<16} {avg_ba:>7.1%} {avg_bm:>9.1%} {avg_loss:>8.4f}  {status}  ({elapsed:.1f}s)')
            results.append((name, avg_ba, avg_bm, avg_loss, status))

        except Exception as e:
            print(f'\r{name:<16} ERROR: {e}')
            results.append((name, 0, 0, 0, f'ERROR: {e}'))

    print('-' * 62)
    mastered = sum(1 for _, ba, _, _, _ in results if ba > 0.95)
    learning = sum(1 for _, ba, _, _, _ in results if 0.55 < ba <= 0.95)
    print(f'Summary: {mastered} mastered, {learning} learning, {len(results)-mastered-learning} baseline')

if __name__ == '__main__':
    main()
