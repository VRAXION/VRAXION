"""Quick direct text input test for GPU model. Loads checkpoint, feeds text, shows predictions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'curriculum_v2', 'checkpoint_latest.pt')
DEVICE = 'cpu'  # Use CPU to avoid GPU contention with training

def load():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = SwarmByteRingModel(
        num_bits=cfg['num_bits'], embedding_dim=cfg['embedding_dim'],
        depth=cfg['depth'], num_beings=cfg['num_beings'],
        num_memory_positions=cfg['seq_len'],
        use_lcx=cfg.get('use_lcx', True), lcx_mode=cfg['lcx_mode'],
        lcx_num_levels=cfg['lcx_num_levels'],
        lcx_level_slots=[int(cfg['lcx_level_slots'])] if not isinstance(cfg['lcx_level_slots'], list) else cfg['lcx_level_slots'],
        lcx_key_dim=cfg['lcx_key_dim'], lcx_top_k=cfg['lcx_top_k'],
        num_pointers=1, attention_radius=cfg['attention_radius'],
        think_ticks=1,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    model.to(DEVICE)
    return model, cfg['seq_len'], ckpt['step']

def bits_to_byte(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= int(b) << (7 - i)
    return v

def safe_chr(b):
    if 32 <= b <= 126: return chr(b)
    if b == 10: return '\\n'
    if b == 9: return '\\t'
    return f'[{b:02x}]'

def test_text(model, text, seq_len, show_n=50):
    raw = text.encode('utf-8', errors='replace')
    if len(raw) < seq_len + 1:
        raw = raw + b' ' * (seq_len + 1 - len(raw))
    arr = np.frombuffer(raw[:seq_len + 1], dtype=np.uint8)
    bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
    x = torch.from_numpy(bits[:seq_len].copy()).unsqueeze(0).to(DEVICE)
    y = torch.from_numpy(bits[1:seq_len + 1].copy()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()

    n = min(show_n, len(text) - 1, seq_len)
    correct = 0

    print(f'\n  Input: "{text[:n+5]}..."')
    print(f'  {"pos":>4} {"in":>5} -> {"expect":>6}  {"got":>6}  {"conf":>5}  {"ok":>3}  bits')
    print(f'  {"-"*65}')

    for t in range(n):
        in_byte = raw[t]
        exp_byte = raw[t + 1]
        pred_byte = bits_to_byte(preds[0, t].cpu())

        bit_marks = []
        confs = []
        for b in range(8):
            p = probs[0, t, b].item()
            actual = int(y[0, t, b].item())
            pred_b = int(preds[0, t, b].item())
            confs.append(p if actual == 1 else (1 - p))
            bit_marks.append('.' if pred_b == actual else 'X')

        avg_conf = sum(confs) / 8
        match = pred_byte == exp_byte
        if match: correct += 1

        in_c = safe_chr(in_byte)
        exp_c = safe_chr(exp_byte)
        pred_c = safe_chr(pred_byte)

        print(f'  {t:>4} {in_c:>5} -> {exp_c:>6}  {pred_c:>6}  {avg_conf:.2f}  {"YES" if match else " no"}  {"".join(bit_marks)}')

    print(f'\n  Byte match: {correct}/{n} = {correct/n:.1%}')
    return correct / n

def test_prefix(model, prefix, seq_len):
    """What does the model predict after a prefix?"""
    raw = prefix.encode('utf-8', errors='replace')
    raw = raw + b' ' * (seq_len + 1 - len(raw))
    arr = np.frombuffer(raw[:seq_len + 1], dtype=np.uint8)
    bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
    x = torch.from_numpy(bits[:seq_len].copy()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out)

    pos = len(prefix) - 1
    pred_bits = (probs[0, pos] > 0.5).float()
    pred_byte = bits_to_byte(pred_bits.cpu())
    pred_char = safe_chr(pred_byte)

    # Show probability for each bit
    bp = [f'{probs[0, pos, b].item():.2f}' for b in range(8)]

    # Generate next 10 characters autoregressively
    gen = []
    cur_bits = bits[:seq_len].copy()
    for _ in range(10):
        x_t = torch.from_numpy(cur_bits.copy()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            o = model(x_t)
            p = torch.sigmoid(o)
        last_pos = min(len(prefix) - 1 + len(gen), seq_len - 1)
        pred = (p[0, last_pos] > 0.5).float().cpu().numpy()
        pb = bits_to_byte(pred)
        gen.append(safe_chr(pb))
        # Shift: push predicted bits into next position
        if last_pos + 1 < seq_len:
            cur_bits[last_pos + 1] = pred
        else:
            break

    return pred_char, bp, ''.join(gen)


def main():
    print('=' * 70)
    print('  DIRECT TEXT INPUT TEST')
    print('=' * 70)

    model, seq_len, step = load()
    print(f'  Loaded step {step}, seq_len={seq_len}, device={DEVICE}')

    # Test 1: Character-by-character predictions
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test.",
        "aaaaaaaaaa bbbbbbbbbb cccccccccc",
    ]

    for text in texts:
        test_text(model, text, seq_len, show_n=25)

    # Test 2: Autoregressive generation from prefixes
    print(f'\n{"="*70}')
    print(f'  AUTOREGRESSIVE GENERATION (10 chars)')
    print(f'{"="*70}')

    prefixes = [
        "The ",
        "Hello ",
        "the ",
        " ",
    ]

    for prefix in prefixes:
        pred_c, bp, gen_text = test_prefix(model, prefix, seq_len)
        print(f'  "{prefix}" -> next="{pred_c}" gen="{gen_text}"')
        print(f'    bit probs: [{" ".join(bp)}]')

    print(f'\n{"="*70}')
    print(f'  Done.')


if __name__ == '__main__':
    main()
