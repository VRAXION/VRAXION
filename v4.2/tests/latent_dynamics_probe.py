"""Latent-dynamics probe for the self-wiring graph.

Tests whether recurrent loops create a useful latent-like internal state-space
rather than only recurrent activity.

Outputs:
- task score / accuracy
- tick-wise state separability on noisy one-hot replicates
- tick-wise effective rank / participation ratio
- feedback-ablation deltas
- perturbation sensitivity by zone and tick
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.graph import SelfWiringGraph
from lib.utils import score_batch


DEFAULT_SEEDS = [0, 42, 123]
DEFAULT_VOCABS = [32, 64]
DEFAULT_TICKS = [4, 8, 16]
DEFAULT_ABLATIONS = ["none", "out_to_compute", "out_to_input", "all_feedback"]
DEFAULT_REPEATS = 8
DEFAULT_NOISE = 0.10
DEFAULT_PERTURB_EPS = 0.05
DEFAULT_MAX_ATT = 2500
DEFAULT_STALE = 1200
DEFAULT_TRAIN_SECONDS = 3.0
DEFAULT_ADD_EVERY = 100
DEFAULT_FRONTLOAD_UNTIL = 200
DEFAULT_FRONTLOAD_EVERY = 10
DEFAULT_CRYSTAL_BUDGET = 600
DEFAULT_CRYSTAL_WINDOW = 120
DEFAULT_CRYSTAL_MIN_RATE = 0.003


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def ensure_results_dir():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def clone_net(net):
    clone = SelfWiringGraph(net.V)
    clone.mask = net.mask.copy()
    clone.resync_alive()
    clone.state = net.state.copy()
    clone.charge = net.charge.copy()
    clone.loss_pct = np.int8(int(net.loss_pct))
    clone.drive = np.int8(int(net.drive))
    return clone


def zone_bounds(net):
    V = net.V
    out_start = net.out_start
    return {
        'input': (0, V),
        'compute': (V, out_start),
        'output': (out_start, out_start + V),
    }


def zone_of(idx, bounds):
    for name, (lo, hi) in bounds.items():
        if lo <= idx < hi:
            return name
    return 'other'


def feedback_edge_count(net):
    bounds = zone_bounds(net)
    order = {'input': 0, 'compute': 1, 'output': 2, 'other': 3}
    count = 0
    for r, c in net.alive:
        if order[zone_of(r, bounds)] > order[zone_of(c, bounds)]:
            count += 1
    return count


def make_probe_worlds(V, repeats=DEFAULT_REPEATS, noise=DEFAULT_NOISE, seed=0):
    rng = np.random.default_rng(seed)
    worlds = np.zeros((V * repeats, V), dtype=np.float32)
    labels = np.repeat(np.arange(V, dtype=np.int32), repeats)
    for cls in range(V):
        base = np.zeros(V, dtype=np.float32)
        base[cls] = 1.0
        for rep in range(repeats):
            idx = cls * repeats + rep
            noise_vec = rng.random(V, dtype=np.float32)
            noise_vec[cls] = 0.0
            noise_sum = float(noise_vec.sum())
            if noise_sum > 0:
                noise_vec /= noise_sum
            w = (1.0 - noise) * base + noise * noise_vec
            worlds[idx] = w
    return worlds, labels


def forward_worlds_trace(net, worlds, ticks=8, perturb=None):
    B, V = worlds.shape
    N = net.N
    charges = np.zeros((B, N), dtype=np.float32)
    acts = np.zeros((B, N), dtype=np.float32)
    retain = float(net.retention)

    trace = {
        'acts': [],
        'charges': [],
        'raw': [],
        'preclip': [],
    }

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds
        if perturb is not None:
            perturb(t, acts, charges)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        preclip = charges.copy()
        charges = np.clip(charges, -1.0, 1.0)

        trace['raw'].append(raw.copy())
        trace['preclip'].append(preclip)
        trace['charges'].append(charges.copy())
        trace['acts'].append(acts.copy())

    trace['logits'] = charges[:, net.out_start:net.out_start + V].copy()
    return trace


def effective_rank_metrics(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return {
            'effective_rank': 0.0,
            'participation_ratio': 0.0,
            'top1_var_ratio': 0.0,
        }
    Xc = X - X.mean(axis=0, keepdims=True)
    if not np.any(Xc):
        return {
            'effective_rank': 0.0,
            'participation_ratio': 0.0,
            'top1_var_ratio': 0.0,
        }
    s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
    eig = (s ** 2) / max(X.shape[0] - 1, 1)
    total = float(eig.sum())
    if total <= 1e-12:
        return {
            'effective_rank': 0.0,
            'participation_ratio': 0.0,
            'top1_var_ratio': 0.0,
        }
    p = eig / total
    p = p[p > 0]
    effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    participation = float((total ** 2) / max(float((eig ** 2).sum()), 1e-12))
    top1 = float(eig[0] / total)
    return {
        'effective_rank': effective_rank,
        'participation_ratio': participation,
        'top1_var_ratio': top1,
    }


def separability_metrics(X, labels, n_classes):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    B, D = X.shape
    counts = np.bincount(labels, minlength=n_classes).astype(np.int32)
    centroids = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        cls = X[labels == c]
        if len(cls) > 0:
            centroids[c] = cls.mean(axis=0)

    global_centroid = X.mean(axis=0, keepdims=True)
    between_scatter = 0.0
    for c in range(n_classes):
        if counts[c] > 0:
            diff = centroids[c] - global_centroid[0]
            between_scatter += counts[c] * float(diff @ diff)
    between_scatter /= max(B, 1)

    within_vals = []
    nearest_other_vals = []
    correct = 0

    for i in range(B):
        own = int(labels[i])
        d2 = np.empty(n_classes, dtype=np.float64)
        for c in range(n_classes):
            centroid = centroids[c]
            if c == own and counts[own] > 1:
                centroid = (centroids[own] * counts[own] - X[i]) / (counts[own] - 1)
            diff = X[i] - centroid
            d2[c] = float(diff @ diff)
        pred = int(np.argmin(d2))
        if pred == own:
            correct += 1
        own_d = np.sqrt(max(d2[own], 0.0))
        other_mask = np.ones(n_classes, dtype=bool)
        other_mask[own] = False
        nearest_other = np.sqrt(max(float(d2[other_mask].min()), 0.0)) if n_classes > 1 else 0.0
        within_vals.append(own_d)
        nearest_other_vals.append(nearest_other)

    within_mean = float(np.mean(within_vals))
    nearest_other_mean = float(np.mean(nearest_other_vals))
    within_scatter = float(np.mean(np.square(within_vals)))
    fisher_ratio = between_scatter / max(within_scatter, 1e-12)
    return {
        'centroid_acc': float(correct / max(B, 1)),
        'within_mean': within_mean,
        'nearest_other_mean': nearest_other_mean,
        'sep_ratio': nearest_other_mean / max(within_mean, 1e-12),
        'fisher_ratio': fisher_ratio,
    }


def compute_slice(net, tensor):
    if net.out_start > net.V:
        return tensor[:, net.V:net.out_start]
    return tensor[:, net.V:]


def analyze_trace(net, trace, labels, n_classes):
    per_tick = []
    for tick_idx, (charges, acts) in enumerate(zip(trace['charges'], trace['acts']), start=1):
        hidden_charges = compute_slice(net, charges)
        hidden_acts = compute_slice(net, acts)
        charge_sep = separability_metrics(hidden_charges, labels, n_classes)
        act_sep = separability_metrics(hidden_acts, labels, n_classes)
        charge_rank = effective_rank_metrics(hidden_charges)
        act_rank = effective_rank_metrics(hidden_acts)
        per_tick.append({
            'tick': tick_idx,
            'hidden_charge_sep_ratio': charge_sep['sep_ratio'],
            'hidden_charge_centroid_acc': charge_sep['centroid_acc'],
            'hidden_charge_fisher_ratio': charge_sep['fisher_ratio'],
            'hidden_charge_effective_rank': charge_rank['effective_rank'],
            'hidden_charge_participation_ratio': charge_rank['participation_ratio'],
            'hidden_charge_top1_var_ratio': charge_rank['top1_var_ratio'],
            'hidden_act_sep_ratio': act_sep['sep_ratio'],
            'hidden_act_centroid_acc': act_sep['centroid_acc'],
            'hidden_act_fisher_ratio': act_sep['fisher_ratio'],
            'hidden_act_effective_rank': act_rank['effective_rank'],
            'hidden_act_participation_ratio': act_rank['participation_ratio'],
            'hidden_act_top1_var_ratio': act_rank['top1_var_ratio'],
        })
    return per_tick


def perturbation_sensitivity(net, worlds, ticks, eps=DEFAULT_PERTURB_EPS):
    baseline = forward_worlds_trace(net, worlds, ticks=ticks)
    base_logits = baseline['logits']
    bounds = zone_bounds(net)
    zone_indices = {
        'input': bounds['input'][0],
        'compute': bounds['compute'][0],
        'output': bounds['output'][0],
    }
    out = {}
    for zone, neuron_idx in zone_indices.items():
        zone_rows = []
        for tick_idx in range(ticks):
            def perturb(t, acts, charges, idx=neuron_idx, target_tick=tick_idx):
                if t == target_tick:
                    charges[:, idx] += np.float32(eps)
            pert = forward_worlds_trace(net, worlds, ticks=ticks, perturb=perturb)
            delta = np.abs(pert['logits'] - base_logits)
            zone_rows.append({
                'tick': tick_idx + 1,
                'mean_abs_logit_delta': float(delta.mean()),
                'max_abs_logit_delta': float(delta.max()),
            })
        out[zone] = zone_rows
    return out


def evaluate_identity(net, targets, V, ticks):
    score, acc = score_batch(net, targets, V, ticks=ticks)
    return float(score), float(acc)


def train_baseline(net, targets, V, ticks, args):
    start = time.time()
    best_sc, best_acc = score_batch(net, targets, V, ticks=ticks)
    kept = 0
    stale = 0
    attempts = 0
    for _ in range(args.max_attempts):
        if time.time() - start >= args.train_seconds:
            break
        attempts += 1
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        sc, acc = score_batch(net, targets, V, ticks=ticks)
        if sc > best_sc:
            best_sc = float(sc)
            best_acc = float(acc)
            kept += 1
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
        if stale >= args.stale_limit or best_sc >= 0.99:
            break
    return {
        'best_score': float(best_sc),
        'best_acc': float(best_acc),
        'kept': int(kept),
        'cycles': 0,
        'attempts': int(attempts),
        'train_seconds': float(time.time() - start),
    }


def apply_ablation(net, mode):
    if mode == 'none':
        return 0
    bounds = zone_bounds(net)
    removed = 0
    for r, c in list(net.alive):
        src = zone_of(r, bounds)
        dst = zone_of(c, bounds)
        kill = False
        if mode == 'out_to_compute':
            kill = (src == 'output' and dst == 'compute')
        elif mode == 'out_to_input':
            kill = (src == 'output' and dst == 'input')
        elif mode == 'all_feedback':
            order = {'input': 0, 'compute': 1, 'output': 2, 'other': 3}
            kill = order[src] > order[dst]
        else:
            raise ValueError(f'Unknown ablation mode: {mode}')
        if kill and net.mask[r, c] != 0:
            net.mask[r, c] = 0
            removed += 1
    if removed:
        net.resync_alive()
    return removed


def summarize_ticks(per_tick):
    if not per_tick:
        return {}
    best_charge_sep = max(per_tick, key=lambda row: row['hidden_charge_sep_ratio'])
    best_act_sep = max(per_tick, key=lambda row: row['hidden_act_sep_ratio'])
    final = per_tick[-1]
    return {
        'best_hidden_charge_sep_ratio': float(best_charge_sep['hidden_charge_sep_ratio']),
        'best_hidden_charge_sep_tick': int(best_charge_sep['tick']),
        'best_hidden_charge_centroid_acc': float(max(row['hidden_charge_centroid_acc'] for row in per_tick)),
        'best_hidden_act_sep_ratio': float(best_act_sep['hidden_act_sep_ratio']),
        'best_hidden_act_sep_tick': int(best_act_sep['tick']),
        'final_hidden_charge_sep_ratio': float(final['hidden_charge_sep_ratio']),
        'final_hidden_act_sep_ratio': float(final['hidden_act_sep_ratio']),
        'final_hidden_charge_effective_rank': float(final['hidden_charge_effective_rank']),
        'final_hidden_act_effective_rank': float(final['hidden_act_effective_rank']),
    }


def run_condition(condition, net, targets, probe_worlds, probe_labels, V, ticks, ablation_mode='none', perturb_eps=DEFAULT_PERTURB_EPS, include_perturb=True):
    work_net = clone_net(net)
    removed_edges = apply_ablation(work_net, ablation_mode)
    score, acc = evaluate_identity(work_net, targets, V, ticks)
    trace = forward_worlds_trace(work_net, probe_worlds, ticks=ticks)
    per_tick = analyze_trace(work_net, trace, probe_labels, n_classes=V)
    perturb = perturbation_sensitivity(work_net, probe_worlds, ticks=ticks, eps=perturb_eps) if include_perturb else {}
    return {
        'condition': condition,
        'ablation_mode': ablation_mode,
        'removed_edges': int(removed_edges),
        'score': float(score),
        'acc': float(acc),
        'conns': int(work_net.count_connections()),
        'feedback_edges': int(feedback_edge_count(work_net)),
        'tick_metrics': per_tick,
        'summary': summarize_ticks(per_tick),
        'perturbation': perturb,
    }


def format_summary_rows(all_results):
    rows = []
    for res in all_results:
        rows.append({
            'V': res['V'],
            'ticks': res['ticks'],
            'seed': res['seed'],
            'condition': res['condition'],
            'score': res['score'],
            'acc': res['acc'],
            'best_hidden_charge_sep_ratio': res['summary']['best_hidden_charge_sep_ratio'],
            'best_hidden_charge_sep_tick': res['summary']['best_hidden_charge_sep_tick'],
            'final_hidden_charge_rank': res['summary']['final_hidden_charge_effective_rank'],
            'feedback_edges': res['feedback_edges'],
            'removed_edges': res['removed_edges'],
        })
    return rows


def print_summary(rows):
    print(f"{'V':>3s} {'t':>3s} {'seed':>4s} {'condition':<16s} {'score':>7s} {'acc':>7s} {'sep*':>7s} {'t*':>3s} {'rank':>7s} {'fb':>5s} {'rm':>5s}")
    print('-' * 78)
    for r in rows:
        print(f"{r['V']:3d} {r['ticks']:3d} {r['seed']:4d} {r['condition']:<16s} {r['score']*100:6.1f}% {r['acc']*100:6.1f}% "
              f"{r['best_hidden_charge_sep_ratio']:6.2f} {r['best_hidden_charge_sep_tick']:3d} {r['final_hidden_charge_rank']:6.2f} {r['feedback_edges']:5d} {r['removed_edges']:5d}")


def aggregate_by_condition(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['V'], row['ticks'], row['condition'])
        grouped[key].append(row)
    agg_rows = []
    for (V, ticks, condition), vals in sorted(grouped.items()):
        agg_rows.append({
            'V': V,
            'ticks': ticks,
            'condition': condition,
            'score_mean': float(np.mean([v['score'] for v in vals])),
            'acc_mean': float(np.mean([v['acc'] for v in vals])),
            'sep_mean': float(np.mean([v['best_hidden_charge_sep_ratio'] for v in vals])),
            'rank_mean': float(np.mean([v['final_hidden_charge_rank'] for v in vals])),
            'fb_mean': float(np.mean([v['feedback_edges'] for v in vals])),
            'rm_mean': float(np.mean([v['removed_edges'] for v in vals])),
        })
    return agg_rows


def print_aggregate(rows):
    print("\nAggregate means")
    print(f"{'V':>3s} {'t':>3s} {'condition':<16s} {'score':>7s} {'acc':>7s} {'sep*':>7s} {'rank':>7s} {'fb':>7s} {'rm':>7s}")
    print('-' * 76)
    for r in rows:
        print(f"{r['V']:3d} {r['ticks']:3d} {r['condition']:<16s} {r['score_mean']*100:6.1f}% {r['acc_mean']*100:6.1f}% "
              f"{r['sep_mean']:6.2f} {r['rank_mean']:6.2f} {r['fb_mean']:6.1f} {r['rm_mean']:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocabs', default=','.join(map(str, DEFAULT_VOCABS)))
    ap.add_argument('--ticks', default=','.join(map(str, DEFAULT_TICKS)))
    ap.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    ap.add_argument('--ablations', default=','.join(DEFAULT_ABLATIONS),
                    help='Comma-separated ablation modes; include none for trained baseline.')
    ap.add_argument('--repeats', type=int, default=DEFAULT_REPEATS)
    ap.add_argument('--noise', type=float, default=DEFAULT_NOISE)
    ap.add_argument('--perturb-eps', type=float, default=DEFAULT_PERTURB_EPS)
    ap.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATT)
    ap.add_argument('--stale-limit', type=int, default=DEFAULT_STALE)
    ap.add_argument('--train-seconds', type=float, default=DEFAULT_TRAIN_SECONDS)
    ap.add_argument('--add-every', type=int, default=DEFAULT_ADD_EVERY)
    ap.add_argument('--frontload-until', type=int, default=DEFAULT_FRONTLOAD_UNTIL)
    ap.add_argument('--frontload-every', type=int, default=DEFAULT_FRONTLOAD_EVERY)
    ap.add_argument('--crystal-budget', type=int, default=DEFAULT_CRYSTAL_BUDGET)
    ap.add_argument('--crystal-window', type=int, default=DEFAULT_CRYSTAL_WINDOW)
    ap.add_argument('--crystal-min-rate', type=float, default=DEFAULT_CRYSTAL_MIN_RATE)
    ap.add_argument('--skip-random-perturb', action='store_true')
    ap.add_argument('--skip-ablation-perturb', action='store_true')
    ap.add_argument('--pilot', action='store_true', help='Run only the first matrix point.')
    ap.add_argument('--tag', default='latent_dynamics_probe')
    args = ap.parse_args()

    vocabs = parse_int_list(args.vocabs)
    tick_list = parse_int_list(args.ticks)
    seeds = parse_int_list(args.seeds)
    ablations = [x.strip() for x in args.ablations.split(',') if x.strip()]

    if args.pilot:
        vocabs = vocabs[:1]
        tick_list = tick_list[:1]
        seeds = seeds[:1]

    all_results = []
    started = time.time()

    for V in vocabs:
        for ticks in tick_list:
            for seed in seeds:
                np.random.seed(seed)
                random.seed(seed)
                targets = np.arange(V, dtype=np.int32)
                np.random.shuffle(targets)
                probe_worlds, probe_labels = make_probe_worlds(V, repeats=args.repeats, noise=args.noise, seed=seed + 12345)

                rand_net = SelfWiringGraph(V)
                random_result = run_condition('random', rand_net, targets, probe_worlds, probe_labels, V, ticks,
                                              perturb_eps=args.perturb_eps,
                                              include_perturb=not args.skip_random_perturb)
                random_result.update({'V': V, 'ticks': ticks, 'seed': seed})
                all_results.append(random_result)

                np.random.seed(seed)
                random.seed(seed)
                trained_net = SelfWiringGraph(V)
                random.seed(seed * 1000 + 1)
                train_meta = train_baseline(trained_net, targets, V, ticks, args)

                baseline_result = run_condition('trained', trained_net, targets, probe_worlds, probe_labels, V, ticks, ablation_mode='none')
                baseline_result.update({'V': V, 'ticks': ticks, 'seed': seed, 'train_meta': train_meta})
                all_results.append(baseline_result)

                for mode in ablations:
                    if mode == 'none':
                        continue
                    ablated = run_condition(f'ablate_{mode}', trained_net, targets, probe_worlds, probe_labels, V, ticks,
                                            ablation_mode=mode,
                                            perturb_eps=args.perturb_eps,
                                            include_perturb=not args.skip_ablation_perturb)
                    ablated.update({'V': V, 'ticks': ticks, 'seed': seed, 'train_meta': train_meta})
                    all_results.append(ablated)

                print(f"Completed V={V} ticks={ticks} seed={seed} | train={train_meta['best_score']*100:.1f}%/{train_meta['best_acc']*100:.1f}% | elapsed={time.time()-started:.1f}s", flush=True)

    rows = format_summary_rows(all_results)
    print()
    print_summary(rows)
    agg = aggregate_by_condition(rows)
    print_aggregate(agg)

    out_dir = ensure_results_dir()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = out_dir / f"{args.tag}_{stamp}"
    payload = {
        'meta': {
            'vocabs': vocabs,
            'ticks': tick_list,
            'seeds': seeds,
            'ablations': ablations,
            'repeats': args.repeats,
            'noise': args.noise,
            'perturb_eps': args.perturb_eps,
            'max_attempts': args.max_attempts,
            'stale_limit': args.stale_limit,
            'train_seconds': args.train_seconds,
            'elapsed_s': float(time.time() - started),
        },
        'rows': rows,
        'aggregate': agg,
        'results': all_results,
    }
    json_path = base.with_suffix('.json')
    json_path.write_text(json.dumps(payload, indent=2, cls=NpEncoder))

    txt_path = base.with_suffix('.txt')
    with txt_path.open('w', encoding='utf-8') as f:
        f.write('Latent Dynamics Probe\n')
        f.write('=' * 80 + '\n')
        f.write(json.dumps(payload['meta'], indent=2, cls=NpEncoder))
        f.write('\n\nAggregate means\n')
        for row in agg:
            f.write(json.dumps(row, cls=NpEncoder) + '\n')
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == '__main__':
    main()
