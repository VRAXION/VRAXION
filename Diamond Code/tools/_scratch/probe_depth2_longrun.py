"""
Probe: Depth=2 long training run (2000 steps) — where does it plateau?

Key question: GPU model (depth=2, D=6180) plateaued at ~76% eval after 1450 steps.
Does the CPU Nano (depth=2, D=2000) also plateau at ~76%, or does it keep improving?

If both plateau at the same level -> architecture limit at depth=2
If Nano keeps improving past 76% -> GPU problem is LCX interference or LR

Eval every 50 steps for clean learning curve. Saves per-eval JSON.
CPU only, 8 threads, safe alongside GPU training.
"""

import sys, os, time, math, random, json

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

STEP_TIMEOUT = 300
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nano')
os.makedirs(LOG_DIR, exist_ok=True)

# Config: depth=2 sweet spot from overnight chain
D = 2000
DEPTH = 2
SEQ_LEN = 256
BATCH = 8
LR = 3e-4
NUM_STEPS = 2000
EVAL_EVERY = 50
EVAL_SAMPLES = 50
RADIUS = 16


def do_eval(model, corpus, chunk_bytes, max_start):
    """Run eval, return metrics dict."""
    model.eval()
    eval_losses, eval_accs = [], []
    eval_per_bit = [[] for _ in range(8)]
    eval_byte_matches = []

    with torch.no_grad():
        for _ in range(EVAL_SAMPLES):
            start = random.randint(0, max_start)
            chunk = corpus[start:start + chunk_bytes]
            arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
            ex = torch.from_numpy(bits[:SEQ_LEN]).unsqueeze(0)
            ey = torch.from_numpy(bits[1:SEQ_LEN + 1]).unsqueeze(0)
            eout = model(ex)
            eloss = nn.functional.binary_cross_entropy_with_logits(eout, ey)
            eval_losses.append(eloss.item())
            epreds = (torch.sigmoid(eout) > 0.5).float()
            eval_accs.append((epreds == ey).float().mean().item())
            for b in range(8):
                bacc = (epreds[0, :, b] == ey[0, :, b]).float().mean().item()
                eval_per_bit[b].append(bacc)
            byte_match = 0
            for t in range(SEQ_LEN):
                pred_byte = 0
                actual_byte = 0
                for b in range(8):
                    pred_byte |= int(epreds[0, t, b].item()) << (7 - b)
                    actual_byte |= int(ey[0, t, b].item()) << (7 - b)
                if pred_byte == actual_byte:
                    byte_match += 1
            eval_byte_matches.append(byte_match / SEQ_LEN)

    model.train()

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_acc = sum(eval_accs) / len(eval_accs)
    eval_bpb = eval_loss / math.log(2) * 8
    per_bit = [sum(b) / len(b) for b in eval_per_bit]
    byte_acc = sum(eval_byte_matches) / len(eval_byte_matches)

    return {
        'eval_loss': round(eval_loss, 4),
        'eval_acc': round(eval_acc, 4),
        'eval_byte_acc': round(byte_acc, 4),
        'eval_bpb': round(eval_bpb, 2),
        'per_bit': [round(p, 3) for p in per_bit],
    }


def main():
    print("=" * 70)
    print("  DEPTH=2 LONG RUN: 2000 steps, eval every 50")
    print(f"  D={D}, depth={DEPTH}, seq={SEQ_LEN}, batch={BATCH}, r={RADIUS}, lr={LR}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load corpus
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus)/1024/1024:.0f} MB")

    # Model
    model = SwarmByteRingModel(
        num_bits=8,
        embedding_dim=D,
        depth=DEPTH,
        num_beings=1,
        num_memory_positions=SEQ_LEN,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=2000,
        lcx_key_dim=max(D // 10, 32),
        lcx_top_k=2,
        num_pointers=1,
        attention_radius=RADIUS,
    )
    model.train()
    model.to('cpu')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  ({n_params * 2 / 1024 / 1024:.1f} MB fp16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    chunk_bytes = SEQ_LEN + 1
    max_start = len(corpus) - chunk_bytes

    # Gzip baseline
    import gzip
    sample = corpus[:50000]
    gzip_bpb = len(gzip.compress(sample, 9)) * 8.0 / len(sample)
    print(f"  Gzip baseline: {gzip_bpb:.2f} bpb")

    # Training
    best_loss = float('inf')
    loss_window = []
    eval_history = []
    train_curve = []
    total_time = 0.0

    log_path = os.path.join(LOG_DIR, 'depth2_longrun.log')
    log_file = open(log_path, 'w')

    for step in range(1, NUM_STEPS + 1):
        t0 = time.time()
        print(f"  starting step {step}...", end='', flush=True)

        # Generate batch
        x_list, y_list = [], []
        for _ in range(BATCH):
            start = random.randint(0, max_start)
            chunk = corpus[start:start + chunk_bytes]
            arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
            x_list.append(bits[:SEQ_LEN])
            y_list.append(bits[1:SEQ_LEN + 1])

        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))

        output = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        total_time += dt
        loss_val = loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(output) > 0.5).float()
            bit_acc = (preds == y).float().mean().item()

        loss_window.append(loss_val)
        if len(loss_window) > 50:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if loss_val < best_loss:
            best_loss = loss_val

        bpb = loss_val / math.log(2) * 8

        # Dashboard log
        line = (f"step {step} | loss {loss_val:.6f} | "
                f"acc={bit_acc:.4f} RD:{dt:.3f} traction={bit_acc:.4f} shard=0/0")
        log_file.write(line + '\n')
        log_file.flush()

        # Track training curve
        train_curve.append({
            'step': step,
            'loss': round(loss_val, 4),
            'acc': round(bit_acc, 4),
            'avg50_loss': round(avg_loss, 4),
        })

        # Eval
        if step % EVAL_EVERY == 0:
            em = do_eval(model, corpus, chunk_bytes, max_start)
            em['step'] = step
            em['train_loss_avg50'] = round(avg_loss, 4)
            em['best_train_loss'] = round(best_loss, 4)
            em['elapsed_s'] = round(total_time, 1)
            eval_history.append(em)

            pb_str = ' '.join(f"{p:.0%}" for p in em['per_bit'])
            content_avg = sum(em['per_bit'][3:]) / 5
            print(f" loss={loss_val:.4f} acc={bit_acc:.3f} bpb={bpb:.2f} "
                  f"avg50={avg_loss:.4f} {dt:.2f}s")
            print(f"    EVAL: loss={em['eval_loss']:.4f} acc={em['eval_acc']:.3f} "
                  f"byte={em['eval_byte_acc']:.3f} content_avg={content_avg:.1%}")
            print(f"    BITS: {pb_str}")

            # Plateau detection: last 5 evals, if acc spread < 1%, flag
            if len(eval_history) >= 5:
                last5 = [e['eval_acc'] for e in eval_history[-5:]]
                spread = max(last5) - min(last5)
                trend = last5[-1] - last5[0]
                if spread < 0.01:
                    print(f"    ** PLATEAU DETECTED: last 5 evals spread={spread:.3f}, trend={trend:+.3f}")
                elif spread < 0.02:
                    print(f"    ** NEAR PLATEAU: last 5 evals spread={spread:.3f}, trend={trend:+.3f}")

        elif step <= 5 or step % 10 == 0:
            print(f" loss={loss_val:.4f} acc={bit_acc:.3f} avg50={avg_loss:.4f} {dt:.2f}s")
        else:
            print(f" {dt:.2f}s")

        if dt > STEP_TIMEOUT:
            print(f"  TIMEOUT at step {step}")
            break
        if loss_val > 10.0 or math.isnan(loss_val):
            print(f"  DIVERGED at step {step}")
            break

    log_file.close()

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  DEPTH=2 LONG RUN -- RESULTS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"\n  Config: D={D}, depth={DEPTH}, seq={SEQ_LEN}, batch={BATCH}, lr={LR}")
    print(f"  Params: {n_params:,}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.1f} hr)")

    # Learning curve table
    print(f"\n  {'step':>6} {'eval_loss':>9} {'eval_acc':>8} {'byte_acc':>8} "
          f"{'bpb':>6} | {'b3':>4} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4} {'content':>7}")
    print(f"  {'-'*70}")
    for e in eval_history:
        content = sum(e['per_bit'][3:]) / 5
        print(f"  {e['step']:>6} {e['eval_loss']:>9.4f} {e['eval_acc']:>8.3f} "
              f"{e['eval_byte_acc']:>8.3f} {e['eval_bpb']:>6.2f} | "
              f"{e['per_bit'][3]:>4.0%} {e['per_bit'][4]:>4.0%} {e['per_bit'][5]:>4.0%} "
              f"{e['per_bit'][6]:>4.0%} {e['per_bit'][7]:>4.0%} "
              f"{content:>7.1%}")

    # Phase analysis
    if len(eval_history) >= 4:
        first_q = eval_history[:len(eval_history)//4]
        last_q = eval_history[-len(eval_history)//4:]
        first_acc = sum(e['eval_acc'] for e in first_q) / len(first_q)
        last_acc = sum(e['eval_acc'] for e in last_q) / len(last_q)
        first_content = sum(sum(e['per_bit'][3:])/5 for e in first_q) / len(first_q)
        last_content = sum(sum(e['per_bit'][3:])/5 for e in last_q) / len(last_q)
        print(f"\n  PHASE ANALYSIS:")
        print(f"    First quarter avg:  acc={first_acc:.3f}  content={first_content:.1%}")
        print(f"    Last quarter avg:   acc={last_acc:.3f}  content={last_content:.1%}")
        print(f"    Delta:              acc={last_acc-first_acc:+.3f}  content={last_content-first_content:+.1%}")
        if last_acc - first_acc > 0.02:
            print(f"    -> STILL LEARNING (depth=2 has room to grow)")
        elif last_acc - first_acc > 0.005:
            print(f"    -> SLOWING DOWN (marginal improvement)")
        else:
            print(f"    -> PLATEAUED (no improvement in second half)")

    # GPU comparison
    print(f"\n  GPU COMPARISON:")
    print(f"    GPU (depth=2, D=6180, step 1450): ~76% eval acc, ~12% byte, LCX ON")
    print(f"    CPU (depth=2, D=2000, step {eval_history[-1]['step'] if eval_history else '?'}): "
          f"{eval_history[-1]['eval_acc']:.1%} eval acc, "
          f"{eval_history[-1]['eval_byte_acc']:.1%} byte, NO LCX")
    if eval_history and eval_history[-1]['eval_acc'] > 0.76:
        print(f"    -> CPU EXCEEDS GPU plateau! GPU problem is likely LCX or LR")
    elif eval_history and eval_history[-1]['eval_acc'] > 0.73:
        print(f"    -> SIMILAR plateau. Architecture limit at depth=2")
    else:
        print(f"    -> CPU below GPU. D matters for content bits")

    # Save
    results_path = os.path.join(LOG_DIR, 'depth2_longrun.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {
                'D': D, 'depth': DEPTH, 'seq_len': SEQ_LEN,
                'batch': BATCH, 'radius': RADIUS, 'lr': LR,
                'steps': NUM_STEPS, 'eval_every': EVAL_EVERY,
            },
            'n_params': n_params,
            'gzip_bpb': gzip_bpb,
            'eval_history': eval_history,
            'total_time': round(total_time, 1),
        }, f, indent=2)
    print(f"\n  Results: {results_path}")
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
