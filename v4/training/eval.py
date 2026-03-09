"""INSTNCT v4 Standalone Evaluation — measure a checkpoint on held-out data.

Loads a training checkpoint and runs forward passes on eval data to compute
loss and accuracy metrics. No backward pass, no optimizer, no training —
purely measurement.

Default device is CPU so eval can run in parallel with GPU training without
competing for VRAM. Use --device cuda if no training is running and you
want faster eval.

Usage:
    python eval.py --checkpoint training_output/ckpt_step_10000.pt --data eval_data/
    python eval.py --checkpoint training_output/ckpt_latest.pt --data eval_data/echo256.traindat
    python eval.py --checkpoint ckpt.pt --data eval_data/ --samples 2048 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────
# eval.py lives in training/ alongside train.py. We need to import from:
#   training/train.py  → ByteDataset, loss functions, accuracy functions, checkpoint loader
#   model/instnct.py   → INSTNCT model class

_TRAINING_DIR = Path(__file__).resolve().parent          # v4/training/
_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model'  # v4/model/

for d in (_TRAINING_DIR, _MODEL_DIR):
    ds = str(d)
    if ds not in sys.path:
        sys.path.insert(0, ds)

from model_factory import build_model_from_spec  # type: ignore[import-not-found]  # noqa: E402
from train import (  # type: ignore[import-not-found]  # noqa: E402
    ByteDataset,
    func_accuracy_bin,
    func_accuracy_emb,
    func_discover_dat,
    func_loadckpt_dct,
    func_maskloss_ce,
    func_maskloss_mse,
)


# ── Eval Loop ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, config, device, n_samples: int):
    """Run evaluation on a dataset and return aggregated metrics.

    Forward-only — no backward, no optimizer. All computation is wrapped
    in torch.no_grad() to save memory and time.

    Args:
        model:     checkpoint-reconstructed model (already on device, in eval mode)
        dataset:   ByteDataset wrapping eval .traindat + .mask files
        config:    dict with 'embed_mode', 'batch_size', 'seq_len', 'eval_seed'
        device:    'cpu' or 'cuda'
        n_samples: total number of sequences to evaluate

    Returns:
        dict with keys: raw_loss, masked_loss, accuracy, masked_acc,
                        n_samples, n_batches, mask_frac
    """
    loss_fn = func_maskloss_ce if config['embed_mode'] else func_maskloss_mse
    acc_fn = func_accuracy_emb if config['embed_mode'] else func_accuracy_bin

    batch_size = config['batch_size']
    # ceiling div — last batch may be smaller, but we sample exactly n_samples
    n_batches = -(-n_samples // batch_size)

    # Exact numerator/denominator accumulators. Do not average batch means:
    # the final batch may be smaller, and mask density may vary across batches.
    raw_loss_num = 0.0
    raw_loss_den = 0.0
    masked_loss_num = 0.0
    masked_loss_den = 0.0
    raw_acc_num = 0.0
    raw_acc_den = 0.0
    masked_acc_num = 0.0
    masked_acc_den = 0.0
    mask_frac_num = 0.0
    mask_frac_den = 0.0

    # Fix dataset RNG for reproducibility — ByteDataset sampling uses its own RNG.
    if hasattr(dataset, 'rng'):
        dataset.rng = np.random.default_rng(config.get('eval_seed', 1337))

    sequential = config.get('sequential', False)
    state = None

    for b in range(n_batches):
        # last batch: clamp to remaining samples
        bs = min(batch_size, n_samples - b * batch_size)
        if sequential:
            xb, yb, mask = dataset.sample_batch_sequential(bs, device)
        else:
            xb, yb, mask = dataset.sample_batch(bs, device)

        pred, state = model(xb, state=state if sequential else None)
        raw_loss, masked_loss = loss_fn(pred, yb, mask)
        raw_acc, masked_acc = acc_fn(pred, yb, mask)

        if config['embed_mode']:
            raw_count = float(yb.numel())
            masked_count = float(mask.sum().item())
        else:
            raw_count = float(pred.numel())
            masked_count = float((mask.sum() * pred.shape[-1]).item())

        mask_count = float(mask.numel())
        raw_loss_num += raw_loss.item() * raw_count
        raw_loss_den += raw_count
        masked_loss_num += masked_loss.item() * masked_count
        masked_loss_den += masked_count
        raw_acc_num += raw_acc * raw_count
        raw_acc_den += raw_count
        masked_acc_num += masked_acc * masked_count
        masked_acc_den += masked_count
        mask_frac_num += float(mask.sum().item())
        mask_frac_den += mask_count

    return {
        'raw_loss':    raw_loss_num / max(raw_loss_den, 1.0),
        'masked_loss': masked_loss_num / max(masked_loss_den, 1.0),
        'accuracy':    raw_acc_num / max(raw_acc_den, 1.0),
        'masked_acc':  masked_acc_num / max(masked_acc_den, 1.0),
        'mask_frac':   mask_frac_num / max(mask_frac_den, 1.0),
        'n_samples':   n_samples,
        'n_batches':   n_batches,
    }


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a VRAXION v4 checkpoint on held-out data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python eval.py --checkpoint training_output/ckpt_step_10000.pt --data eval_data/
  python eval.py --checkpoint ckpt.pt --data eval_data/echo256.traindat --samples 2048
  python eval.py --checkpoint ckpt.pt --data eval_data/ --device cuda""",
    )

    parser.add_argument('--checkpoint', required=True,
                        help='path to training checkpoint (.pt file)')
    parser.add_argument('--data', required=True,
                        help='eval data: directory of .traindat files or a single .traindat file')
    parser.add_argument('--samples', type=int, default=1024,
                        help='number of sequences to evaluate (default: 1024)')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size (default: from checkpoint config)')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='sequence length override (default: from checkpoint config)')
    parser.add_argument('--device', default='cpu',
                        help='device for eval (default: cpu — runs alongside GPU training)')

    args = parser.parse_args()

    # ── Resolve paths from v4/ root (same convention as train.py) ──
    v4_root = Path(__file__).resolve().parent.parent

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = v4_root / ckpt_path

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = v4_root / data_path

    # ── Load checkpoint ──
    print(f'VRAXION v4 -- Standalone Evaluation')
    print(f'{"=" * 48}')

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[BOOT] Device     {device}')
    print(f'[BOOT] Checkpoint {ckpt_path}')

    ckpt = func_loadckpt_dct(str(ckpt_path), device)
    train_cfg = ckpt['train_config_resolved']
    model_rec = ckpt['model']
    step = ckpt['step']
    best_loss = ckpt['best_loss']

    print(f'[BOOT] Step       {step}')
    print(f'[BOOT] Best loss  {best_loss:.6f}')

    embed_mode = train_cfg.get('embed_mode', False)
    seq_len = args.seq_len if args.seq_len is not None else train_cfg.get('seq_len', 128)
    batch_size = args.batch if args.batch is not None else train_cfg.get('batch_size', 32)
    mode_str = 'embed (256 tokens, CE loss)' if embed_mode else 'binary (8-bit, MSE loss)'
    print(f'[BOOT] Model      {model_rec.get("type", "unknown")}')
    print(f'[BOOT] Mode       {mode_str}')
    print(f'[BOOT] Seq len    {seq_len}')
    print(f'[BOOT] Batch      {batch_size}')

    # ── Build model ──
    model = build_model_from_spec(model_rec, device=device)
    _missing, _unexpected = model.load_state_dict(model_rec['state_dict'], strict=False)
    if _missing:
        print(f'[BOOT] New params (init from scratch): {_missing}')
    if _unexpected:
        print(f'[BOOT] Dropped params (no longer in model): {_unexpected}')
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[BOOT] Params     {n_params:,}')

    # ── Load eval data ──
    # func_discover_dat handles: single .traindat file, flat dir, or dir with subdirs.
    # Returns list of (traindat_path, mask_path, byte_count) tuples.
    file_pairs = func_discover_dat(str(data_path))
    eval_seed = int(ckpt.get('rng_state', {}).get('eval_seed', train_cfg.get('eval_seed', 1337)))
    dataset = ByteDataset(file_pairs, seq_len, embed_mode, seed=eval_seed)
    print(f'[BOOT] Eval data  {len(file_pairs)} file(s), '
          f'{dataset.total_bytes / 1024**2:.1f} MB, '
          f'{dataset.n_samples:,} samples')

    n_samples = min(args.samples, dataset.n_samples)
    if n_samples < args.samples:
        print(f'[WARN] Requested {args.samples} samples but only {dataset.n_samples} available')

    print(f'[BOOT] Eval size  {n_samples} samples')

    # ── Run eval ──
    sequential = train_cfg.get('sequential', False)
    eval_config = {
        'embed_mode': embed_mode,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'sequential': sequential,
        'eval_seed': eval_seed,
    }

    print(f'\nEvaluating ...')
    t0 = time.perf_counter()
    metrics = evaluate(model, dataset, eval_config, device, n_samples)
    elapsed = time.perf_counter() - t0

    # ── Results ──
    print()
    raw_bpc = metrics['raw_loss'] * 1.4427      # nats -> bits (log2(e))
    masked_bpc = metrics['masked_loss'] * 1.4427
    print(f'[RESULT] raw_loss    {metrics["raw_loss"]:.6f}  (bpc={raw_bpc:.3f})')
    print(f'[RESULT] masked_loss {metrics["masked_loss"]:.6f}  (bpc={masked_bpc:.3f})')
    print(f'[RESULT] accuracy    {metrics["accuracy"]:.4f}  ({metrics["accuracy"]*100:.2f}%)')
    print(f'[RESULT] masked_acc  {metrics["masked_acc"]:.4f}  ({metrics["masked_acc"]*100:.2f}%)')
    print(f'[RESULT] mask_frac   {metrics["mask_frac"]:.4f}')
    print(f'[RESULT] samples     {metrics["n_samples"]}')
    print(f'[RESULT] batches     {metrics["n_batches"]}')
    print(f'[RESULT] wall_time   {elapsed:.1f}s')
    if elapsed > 0:
        print(f'[RESULT] throughput  {metrics["n_samples"] / elapsed:.0f} samples/s')


# ── Entry Point ───────────────────────────────────────────────

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INTERRUPTED] Eval stopped by user.')
        sys.exit(0)
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
