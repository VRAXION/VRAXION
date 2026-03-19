"""Quick loop-attribution probe.

Goal: inspect what the most helpful short active loops seem to do on the
strongest recurrent setting found so far: V=32, ticks=8.

Method:
- train a quick baseline
- trace source activations over ticks
- estimate active edge flow
- enumerate short directed cycles (2- and 3-cycles) in the active subgraph
- classify cycles by sign product and zone pattern
- ablate top positive vs top negative cycles and compare score drop
"""
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.graph import SelfWiringGraph
from lib.utils import score_batch

V = 32
TICKS = 8
SEEDS = [0, 42, 123]
TRAIN_SECONDS = 3.0
PROBE_REPEATS = 8
PROBE_NOISE = 0.10
ACTIVE_Q = 0.70
TOP_K = 12


def clone_net(net):
    clone = SelfWiringGraph(net.V)
    clone.mask = net.mask.copy()
    clone.resync_alive()
    clone.state = net.state.copy()
    clone.charge = net.charge.copy()
    clone.loss_pct = np.int8(int(net.loss_pct))
    clone.drive = np.int8(int(net.drive))
    return clone


def make_probe_worlds(V, repeats=PROBE_REPEATS, noise=PROBE_NOISE, seed=0):
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
            worlds[idx] = (1.0 - noise) * base + noise * noise_vec
    return worlds, labels


def zone_bounds(net):
    return {
        'input': (0, net.V),
        'compute': (net.V, net.out_start),
        'output': (net.out_start, net.out_start + net.V),
    }


def zone_of(idx, bounds):
    for name, (lo, hi) in bounds.items():
        if lo <= idx < hi:
            return name
    return 'other'


def is_feedback_edge(r, c, bounds):
    order = {'input': 0, 'compute': 1, 'output': 2, 'other': 3}
    return order[zone_of(r, bounds)] > order[zone_of(c, bounds)]


def evaluate(net, targets, ticks=TICKS):
    sc, acc = score_batch(net, targets, net.V, ticks=ticks)
    return float(sc), float(acc)


def train_quick(seed, V=V, ticks=TICKS, train_seconds=TRAIN_SECONDS):
    np.random.seed(seed)
    random.seed(seed)
    targets = np.arange(V, dtype=np.int32)
    np.random.shuffle(targets)
    net = SelfWiringGraph(V)
    best_sc, best_acc = evaluate(net, targets, ticks)
    stale = 0
    attempts = 0
    t0 = time.time()
    while time.time() - t0 < train_seconds:
        attempts += 1
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        sc, acc = evaluate(net, targets, ticks)
        if sc > best_sc:
            best_sc, best_acc = sc, acc
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
        if stale >= 1200 or best_sc >= 0.99:
            break
    return net, targets, {'best_score': best_sc, 'best_acc': best_acc, 'attempts': attempts, 'train_seconds': time.time() - t0}


def forward_trace(net, worlds, ticks=TICKS):
    B, V = worlds.shape
    N = net.N
    charges = np.zeros((B, N), dtype=np.float32)
    acts = np.zeros((B, N), dtype=np.float32)
    retain = float(net.retention)
    trace = {'preacts': [], 'acts': [], 'charges': []}
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds
        preacts = acts.copy()
        raw = preacts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
        trace['preacts'].append(preacts)
        trace['acts'].append(acts.copy())
        trace['charges'].append(charges.copy())
    trace['logits'] = charges[:, net.out_start:net.out_start + V]
    return trace


def edge_flow_matrix(net, trace):
    N = net.N
    flow = np.zeros((N, N), dtype=np.float64)
    abs_mask = np.abs(net.mask).astype(np.float64)
    for preacts in trace['preacts']:
        src_mag = np.mean(np.abs(preacts), axis=0).astype(np.float64)
        flow += src_mag[:, None] * abs_mask
    flow /= max(len(trace['preacts']), 1)
    flow[net.mask == 0] = 0.0
    return flow


def active_edge_mask(net, flow, q=ACTIVE_Q):
    nz = flow[net.mask != 0]
    thresh = float(np.quantile(nz, q)) if len(nz) else 0.0
    active = (net.mask != 0) & (flow >= thresh)
    return active, thresh


def canonical_cycle(nodes):
    rots = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
    return min(rots)


def enumerate_cycles(net, flow, active):
    N = net.N
    bounds = zone_bounds(net)
    adj = {i: np.where(active[i])[0].tolist() for i in range(N)}
    cycles2 = []
    seen2 = set()
    for a in range(N):
        for b in adj[a]:
            if a < b and active[b, a]:
                cyc = (a, b)
                seen2.add(cyc)
                edges = [(a, b), (b, a)]
                sign_prod = int(np.sign(net.mask[a, b] * net.mask[b, a]))
                strength = float(min(flow[a, b], flow[b, a]))
                cycles2.append({
                    'nodes': cyc,
                    'edges': edges,
                    'length': 2,
                    'sign_product': sign_prod,
                    'strength': strength,
                    'zone_pattern': f"{zone_of(a, bounds)}->{zone_of(b, bounds)}->{zone_of(a, bounds)}",
                })

    cycles3 = []
    seen3 = set()
    for a in range(N):
        for b in adj[a]:
            if b == a:
                continue
            for c in adj[b]:
                if c in (a, b):
                    continue
                if active[c, a]:
                    canon = canonical_cycle([a, b, c])
                    if canon in seen3:
                        continue
                    seen3.add(canon)
                    edges = [(a, b), (b, c), (c, a)]
                    sign_prod = int(np.sign(net.mask[a, b] * net.mask[b, c] * net.mask[c, a]))
                    strength = float(min(flow[a, b], flow[b, c], flow[c, a]))
                    cycles3.append({
                        'nodes': canon,
                        'edges': edges,
                        'length': 3,
                        'sign_product': sign_prod,
                        'strength': strength,
                        'zone_pattern': f"{zone_of(a, bounds)}->{zone_of(b, bounds)}->{zone_of(c, bounds)}->{zone_of(a, bounds)}",
                    })
    cycles2.sort(key=lambda x: x['strength'], reverse=True)
    cycles3.sort(key=lambda x: x['strength'], reverse=True)
    return cycles2, cycles3


def ablate_edges(net, edges, targets, ticks=TICKS):
    clone = clone_net(net)
    removed = 0
    for r, c in edges:
        if clone.mask[r, c] != 0:
            clone.mask[r, c] = 0
            removed += 1
    if removed:
        clone.resync_alive()
    sc, acc = evaluate(clone, targets, ticks)
    return {'score': sc, 'acc': acc, 'removed_edges': removed}


def summarize_cycle_family(cycles):
    if not cycles:
        return {'count': 0, 'mean_strength': 0.0, 'patterns': {}}
    return {
        'count': len(cycles),
        'mean_strength': float(np.mean([c['strength'] for c in cycles])),
        'patterns': dict(Counter(c['zone_pattern'] for c in cycles).most_common(5)),
    }


def top_cycle_edges(cycles, top_k=TOP_K):
    edges = []
    for cyc in cycles[:top_k]:
        edges.extend(cyc['edges'])
    uniq = list(dict.fromkeys(edges))
    return uniq


def main():
    out_dir = Path(__file__).resolve().parent / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    seed_results = []
    for seed in SEEDS:
        net, targets, train_meta = train_quick(seed)
        base_sc, base_acc = evaluate(net, targets)
        worlds, labels = make_probe_worlds(V, seed=seed + 12345)
        trace = forward_trace(net, worlds)
        flow = edge_flow_matrix(net, trace)
        active, thresh = active_edge_mask(net, flow)
        cycles2, cycles3 = enumerate_cycles(net, flow, active)
        bounds = zone_bounds(net)
        all_cycles = cycles2 + cycles3
        pos_cycles = [c for c in all_cycles if c['sign_product'] > 0]
        neg_cycles = [c for c in all_cycles if c['sign_product'] < 0]
        pos_fb_cycles = [c for c in pos_cycles if any(is_feedback_edge(r, c2, bounds) for r, c2 in c['edges'])]
        neg_fb_cycles = [c for c in neg_cycles if any(is_feedback_edge(r, c2, bounds) for r, c2 in c['edges'])]
        pos_edges = top_cycle_edges(sorted(pos_fb_cycles, key=lambda x: x['strength'], reverse=True))
        neg_edges = top_cycle_edges(sorted(neg_fb_cycles, key=lambda x: x['strength'], reverse=True))
        pos_abl = ablate_edges(net, pos_edges, targets)
        neg_abl = ablate_edges(net, neg_edges, targets)
        seed_results.append({
            'seed': seed,
            'train_meta': train_meta,
            'baseline': {'score': base_sc, 'acc': base_acc},
            'active_threshold': thresh,
            'cycles_2': summarize_cycle_family(cycles2),
            'cycles_3': summarize_cycle_family(cycles3),
            'positive_cycles': summarize_cycle_family(pos_cycles),
            'negative_cycles': summarize_cycle_family(neg_cycles),
            'positive_feedback_cycles': summarize_cycle_family(pos_fb_cycles),
            'negative_feedback_cycles': summarize_cycle_family(neg_fb_cycles),
            'top_positive_examples': pos_fb_cycles[:5],
            'top_negative_examples': neg_fb_cycles[:5],
            'ablate_positive_top': {
                **pos_abl,
                'score_drop': base_sc - pos_abl['score'],
                'acc_drop': base_acc - pos_abl['acc'],
                'edge_count': len(pos_edges),
            },
            'ablate_negative_top': {
                **neg_abl,
                'score_drop': base_sc - neg_abl['score'],
                'acc_drop': base_acc - neg_abl['acc'],
                'edge_count': len(neg_edges),
            },
        })
        print(f"seed={seed} base={base_sc*100:.1f}%/{base_acc*100:.1f}% | pos_fb_drop={100*(base_sc-pos_abl['score']):.2f}pp neg_fb_drop={100*(base_sc-neg_abl['score']):.2f}pp | pos_fb={len(pos_fb_cycles)} neg_fb={len(neg_fb_cycles)}", flush=True)

    agg = {
        'baseline_score_mean': float(np.mean([r['baseline']['score'] for r in seed_results])),
        'baseline_acc_mean': float(np.mean([r['baseline']['acc'] for r in seed_results])),
        'positive_feedback_cycle_count_mean': float(np.mean([r['positive_feedback_cycles']['count'] for r in seed_results])),
        'negative_feedback_cycle_count_mean': float(np.mean([r['negative_feedback_cycles']['count'] for r in seed_results])),
        'positive_feedback_cycle_strength_mean': float(np.mean([r['positive_feedback_cycles']['mean_strength'] for r in seed_results])),
        'negative_feedback_cycle_strength_mean': float(np.mean([r['negative_feedback_cycles']['mean_strength'] for r in seed_results])),
        'ablate_positive_score_drop_mean': float(np.mean([r['ablate_positive_top']['score_drop'] for r in seed_results])),
        'ablate_negative_score_drop_mean': float(np.mean([r['ablate_negative_top']['score_drop'] for r in seed_results])),
        'ablate_positive_acc_drop_mean': float(np.mean([r['ablate_positive_top']['acc_drop'] for r in seed_results])),
        'ablate_negative_acc_drop_mean': float(np.mean([r['ablate_negative_top']['acc_drop'] for r in seed_results])),
        'top_positive_zone_patterns': dict(Counter(
            p for r in seed_results for p in r['positive_feedback_cycles']['patterns'].keys()
        ).most_common(8)),
        'top_negative_zone_patterns': dict(Counter(
            p for r in seed_results for p in r['negative_feedback_cycles']['patterns'].keys()
        ).most_common(8)),
    }

    payload = {
        'meta': {
            'V': V,
            'ticks': TICKS,
            'seeds': SEEDS,
            'train_seconds': TRAIN_SECONDS,
            'probe_repeats': PROBE_REPEATS,
            'probe_noise': PROBE_NOISE,
            'active_q': ACTIVE_Q,
            'top_k_cycles': TOP_K,
        },
        'aggregate': agg,
        'seed_results': seed_results,
    }

    json_path = out_dir / f'loop_attribution_probe_{stamp}.json'
    txt_path = out_dir / f'loop_attribution_probe_{stamp}.txt'
    json_path.write_text(json.dumps(payload, indent=2))
    with txt_path.open('w', encoding='utf-8') as f:
        f.write('Loop Attribution Probe\n')
        f.write('=' * 80 + '\n')
        f.write(json.dumps(payload['meta'], indent=2))
        f.write('\n\nAggregate\n')
        f.write(json.dumps(payload['aggregate'], indent=2))
    print('\nAggregate:')
    print(json.dumps(agg, indent=2))
    print(f'\nSaved: {json_path}')
    print(f'Saved: {txt_path}')


if __name__ == '__main__':
    main()
