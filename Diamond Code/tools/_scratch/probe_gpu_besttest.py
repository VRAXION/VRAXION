"""
GPU Best Test: Load latest checkpoint, run detailed eval + direct text input.
Runs alongside training (batch=1, minimal VRAM).
"""
import sys, os, time, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'curriculum_v2')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(ckpt_path):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = SwarmByteRingModel(
        num_bits=cfg['num_bits'],
        embedding_dim=cfg['embedding_dim'],
        depth=cfg['depth'],
        num_beings=cfg['num_beings'],
        num_memory_positions=cfg['seq_len'],
        use_lcx=cfg.get('use_lcx', True),
        lcx_mode=cfg['lcx_mode'],
        lcx_num_levels=cfg['lcx_num_levels'],
        lcx_level_slots=[int(cfg['lcx_level_slots'])] if not isinstance(cfg['lcx_level_slots'], list) else cfg['lcx_level_slots'],
        lcx_key_dim=cfg['lcx_key_dim'],
        lcx_top_k=cfg['lcx_top_k'],
        num_pointers=1,
        attention_radius=cfg['attention_radius'],
        think_ticks=1,  # Training runs at tt=1 (Beta stage)
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    model.to(DEVICE)
    return model, ckpt

def swap_lcx(model, lcx_state):
    """Swap LCX keys/values/heat/valid from a snapshot."""
    sd = model.state_dict()
    sd['lcx_keys_0'] = lcx_state['L0_keys']
    sd['lcx_values_0'] = lcx_state['L0_values']
    sd['lcx_heat_0'] = lcx_state['L0_heat']
    sd['lcx_valid_0'] = lcx_state['L0_valid']
    model.load_state_dict(sd, strict=False)

def text_to_bits(text, seq_len):
    """Convert text to bit tensor [1, seq_len, 8] using binary_bits encoding."""
    raw = text.encode('utf-8', errors='replace')
    if len(raw) < seq_len + 1:
        raw = raw + b'\x00' * (seq_len + 1 - len(raw))
    arr = np.frombuffer(raw[:seq_len + 1], dtype=np.uint8)
    # Reshape [seq_len+1] -> [seq_len+1, 1] then unpackbits on axis=1 -> [seq_len+1, 8]
    bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
    x = torch.from_numpy(bits[:seq_len].copy()).unsqueeze(0).to(DEVICE)
    y = torch.from_numpy(bits[1:seq_len + 1].copy()).unsqueeze(0).to(DEVICE)
    return x, y, raw

def bits_to_byte(bit_array):
    """Convert 8-element bit array to byte value."""
    val = 0
    for i, b in enumerate(bit_array):
        val |= int(b) << (7 - i)
    return val

def byte_to_char(b):
    """Safe byte to printable char."""
    if 32 <= b <= 126:
        return chr(b)
    elif b == 10:
        return '\\n'
    elif b == 13:
        return '\\r'
    elif b == 9:
        return '\\t'
    else:
        return f'x{b:02x}'

def run_eval(model, corpus, seq_len, n_samples=200):
    """Detailed eval with per-bit and per-position analysis."""
    chunk_bytes = seq_len + 1
    max_start = len(corpus) - chunk_bytes

    all_losses, all_accs = [], []
    per_bit_correct = [0] * 8
    per_bit_total = [0] * 8
    byte_matches = 0
    byte_total = 0

    # Per-position accuracy (first 32 positions)
    pos_correct = [0] * min(32, seq_len)
    pos_total = [0] * min(32, seq_len)

    with torch.no_grad():
        for i in range(n_samples):
            start = random.randint(0, max_start)
            chunk = corpus[start:start + chunk_bytes]
            arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)

            x = torch.from_numpy(bits[:seq_len]).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(bits[1:seq_len + 1]).unsqueeze(0).to(DEVICE)

            out = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y)
            all_losses.append(loss.item())

            preds = (torch.sigmoid(out) > 0.5).float()
            all_accs.append((preds == y).float().mean().item())

            # Per-bit
            for b in range(8):
                correct = (preds[0, :, b] == y[0, :, b]).float().sum().item()
                per_bit_correct[b] += correct
                per_bit_total[b] += seq_len

            # Byte match
            for t in range(seq_len):
                pred_byte = bits_to_byte(preds[0, t].cpu())
                actual_byte = bits_to_byte(y[0, t].cpu())
                if pred_byte == actual_byte:
                    byte_matches += 1
                byte_total += 1

                # Per-position (first 32)
                if t < len(pos_correct):
                    if pred_byte == actual_byte:
                        pos_correct[t] += 1
                    pos_total[t] += 1

            if (i + 1) % 50 == 0:
                print(f'  eval {i+1}/{n_samples}...', flush=True)

    results = {
        'loss': sum(all_losses) / len(all_losses),
        'acc': sum(all_accs) / len(all_accs),
        'byte_acc': byte_matches / byte_total,
        'bpb': sum(all_losses) / len(all_losses) / math.log(2) * 8,
        'per_bit': [per_bit_correct[b] / per_bit_total[b] for b in range(8)],
        'pos_acc': [pos_correct[t] / pos_total[t] for t in range(len(pos_correct))],
    }
    return results

def test_direct_input(model, text, seq_len=192, show_n=60):
    """Feed direct text, show character-by-character predictions."""
    x, y, raw = text_to_bits(text, seq_len)

    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()

    # Show character by character
    n_show = min(show_n, len(text) - 1, seq_len)

    print(f'\n  Input:  "{text[:n_show+1]}"')
    print(f'  {"pos":>4} {"in":>4} {"expect":>7} {"predict":>8} {"conf":>5} {"match":>5} {"bit_detail":>20}')
    print(f'  {"-"*60}')

    correct_bytes = 0
    total_bytes = 0

    for t in range(n_show):
        in_byte = raw[t]
        expect_byte = raw[t + 1]
        pred_byte = bits_to_byte(preds[0, t].cpu())

        # Confidence: average probability of correct bits
        bit_confs = []
        bit_detail = []
        for b in range(8):
            p = probs[0, t, b].item()
            actual = int(y[0, t, b].item())
            pred_b = int(preds[0, t, b].item())
            conf = p if actual == 1 else (1 - p)
            bit_confs.append(conf)
            if pred_b == actual:
                bit_detail.append('.')
            else:
                bit_detail.append('X')

        avg_conf = sum(bit_confs) / len(bit_confs)
        match = pred_byte == expect_byte
        if match:
            correct_bytes += 1
        total_bytes += 1

        in_c = byte_to_char(in_byte)
        exp_c = byte_to_char(expect_byte)
        pred_c = byte_to_char(pred_byte)
        match_s = 'YES' if match else 'no'

        print(f'  {t:>4} {in_c:>4} -> {exp_c:>4}  got {pred_c:>4}  {avg_conf:.2f}  {match_s:>5}  {"".join(bit_detail)}')

    print(f'\n  Byte accuracy on this input: {correct_bytes}/{total_bytes} = {correct_bytes/total_bytes:.1%}')
    return correct_bytes / total_bytes if total_bytes > 0 else 0


def main():
    print('=' * 70)
    print('  GPU BEST TEST — Checkpoint Evaluation')
    print(f'  Device: {DEVICE}')
    print(f'  Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # Load model from latest checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_latest.pt')
    print(f'\n  Loading checkpoint: {ckpt_path}')
    t0 = time.time()
    model, ckpt = load_model(ckpt_path)
    print(f'  Loaded in {time.time()-t0:.1f}s (step {ckpt["step"]})')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    seq_len = ckpt['config']['seq_len']
    print(f'  Config: D={ckpt["config"]["embedding_dim"]}, depth={ckpt["config"]["depth"]}, seq={seq_len}')

    # Load corpus
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f'  Corpus: {len(corpus)/1024/1024:.0f} MB')

    # ===== TEST 1: Detailed eval with GOLDEN LCX (from checkpoint) =====
    print(f'\n{"="*70}')
    print(f'  TEST 1: Eval with GOLDEN LCX (checkpoint state)')
    print(f'{"="*70}')

    t0 = time.time()
    results = run_eval(model, corpus, seq_len, n_samples=50)
    dt = time.time() - t0

    print(f'\n  Loss:     {results["loss"]:.4f}')
    print(f'  Bit Acc:  {results["acc"]:.3f} ({results["acc"]:.1%})')
    print(f'  Byte Acc: {results["byte_acc"]:.3f} ({results["byte_acc"]:.1%})')
    print(f'  BPB:      {results["bpb"]:.2f}')
    print(f'  Time:     {dt:.1f}s ({dt/200:.2f}s/sample)')

    # Per-bit breakdown
    print(f'\n  Per-bit accuracy:')
    labels = ['MSB(0)', 'bit1', 'bit2', 'bit3', 'bit4', 'bit5', 'bit6', 'LSB(7)']
    meanings = ['sign/high', 'upper', 'case/num', 'group', 'char-hi', 'char', 'char', 'char-lo']
    for b in range(8):
        acc = results['per_bit'][b]
        bar = '=' * int(acc * 30) + '.' * (30 - int(acc * 30))
        print(f'    bit{b} ({meanings[b]:>8}): {acc:6.1%}  [{bar}]')

    # Structure vs Content
    struct_acc = sum(results['per_bit'][:3]) / 3
    content_acc = sum(results['per_bit'][3:]) / 5
    print(f'\n  Structure bits (0-2): {struct_acc:.1%}')
    print(f'  Content bits  (3-7): {content_acc:.1%}')
    print(f'  Gap: {struct_acc - content_acc:.1%}')

    # Per-position byte accuracy (first 32)
    print(f'\n  Per-position byte accuracy (first 32 positions):')
    pos_line = '  '
    for t in range(0, min(32, len(results['pos_acc'])), 4):
        chunk = results['pos_acc'][t:t+4]
        pos_line += '  '.join(f'{a:.0%}' for a in chunk) + '  |  '
    print(pos_line)

    # ===== TEST 2: Load scratch snapshot and compare =====
    snap_dir = os.path.join(CKPT_DIR, 'sleep_snapshots')
    snaps = sorted([f for f in os.listdir(snap_dir) if f.startswith('snap_')])
    if snaps:
        latest_snap = snaps[-1]
        print(f'\n{"="*70}')
        print(f'  TEST 2: Eval with SCRATCH LCX ({latest_snap})')
        print(f'{"="*70}')

        snap = torch.load(os.path.join(snap_dir, latest_snap), map_location='cpu', weights_only=False)
        print(f'  Snapshot step: {snap["step"]}, loss: {snap["loss"]:.4f}, acc: {snap["bit_acc"]:.3f}')

        # Swap LCX
        swap_lcx(model, snap['lcx'])
        model.to(DEVICE)

        t0 = time.time()
        results2 = run_eval(model, corpus, seq_len, n_samples=30)
        dt = time.time() - t0

        print(f'\n  Loss:     {results2["loss"]:.4f} (golden: {results["loss"]:.4f}, delta: {results2["loss"]-results["loss"]:+.4f})')
        print(f'  Bit Acc:  {results2["acc"]:.3f} (golden: {results["acc"]:.3f}, delta: {results2["acc"]-results["acc"]:+.3f})')
        print(f'  Byte Acc: {results2["byte_acc"]:.3f} (golden: {results["byte_acc"]:.3f}, delta: {results2["byte_acc"]-results["byte_acc"]:+.3f})')

        for b in range(8):
            d = results2['per_bit'][b] - results['per_bit'][b]
            print(f'    bit{b}: {results2["per_bit"][b]:.1%} ({d:+.1%})')

        # Restore golden LCX for direct input tests
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.to(DEVICE)

    # ===== TEST 3: Direct text inputs =====
    print(f'\n{"="*70}')
    print(f'  TEST 3: Direct Text Input')
    print(f'{"="*70}')

    test_texts = [
        "The quick brown fox jumps over the lazy dog. The sun was setting behind the mountains, casting long shadows across the valley.",
        "Hello, my name is VRAXION. I am a language model trained on text data. I can predict the next character in a sequence.",
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 The End.",
        "In 2026, artificial intelligence continued to advance rapidly. New models were being developed that could understand and generate human language.",
        "aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd eeeeeeeeee",
    ]

    for text in test_texts:
        print(f'\n  {"-"*60}')
        test_direct_input(model, text, seq_len, show_n=50)

    # ===== TEST 4: What does it "think" comes after common prefixes? =====
    print(f'\n{"="*70}')
    print(f'  TEST 4: Next-character predictions for common prefixes')
    print(f'{"="*70}')

    prefixes = [
        "The ",
        "Hello ",
        "In the ",
        "a",
        "the",
        " ",
        "e",
        "t",
        "1234",
        "http",
    ]

    for prefix in prefixes:
        x, y, raw = text_to_bits(prefix + '\x00' * 200, seq_len)
        with torch.no_grad():
            out = model(x)
            probs = torch.sigmoid(out)

        # Get prediction at the last real character position
        pos = len(prefix) - 1
        pred_bits = (probs[0, pos] > 0.5).float()
        pred_byte = bits_to_byte(pred_bits.cpu())
        pred_char = byte_to_char(pred_byte)

        # Top confidence bits
        bit_probs = [probs[0, pos, b].item() for b in range(8)]
        bit_str = ' '.join(f'{p:.2f}' for p in bit_probs)

        # Also show what probabilities look like for a few alternative bytes
        # by checking which bytes are "close" to the prediction

        print(f'  "{prefix}" -> predicted next: "{pred_char}" (0x{pred_byte:02x})')
        print(f'    bit probs: [{bit_str}]')

    print(f'\n{"="*70}')
    print(f'  Done: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
