"""
Deep structural evolution analysis
====================================
Train a network and capture snapshots every 500 steps.
Compare snapshots to see WHAT changes structurally:
- Which edges appear/disappear?
- Do hubs form? Which neurons become hubs?
- Does the I/O zone wiring change differently from internal?
- Reciprocal connections (A→B AND B→A)?
- Clustering / community structure?
- Path length evolution?
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']
WAVE_LUT = SelfWiringGraph.WAVE_LUT

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

_bp_out=None;_all_data=None;_bigram=None;_pol=None;_channel=None

def init_w(bpo,data,bg,pol,ch):
    global _bp_out,_all_data,_bigram,_pol,_channel
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_channel=ch

def _eval_bigram(mask, theta, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                channel=_channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,channel,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    elif pt=='flip':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)];nc2=rng.randint(0,H-1)
        if nc2==r or nc2==c or mask[r,nc2]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc2]=True
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,seqs);new=_eval_bigram(nm,nt,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None}

def analyze_structure(mask, polarity, theta, channel, label=""):
    """Deep structural analysis of the current network."""
    H = mask.shape[0]
    in_deg = mask.sum(axis=0)   # incoming
    out_deg = mask.sum(axis=1)  # outgoing
    total_deg = in_deg + out_deg

    print(f"\n  {'='*60}")
    print(f"  STRUCTURAL ANALYSIS: {label}")
    print(f"  {'='*60}")

    # Basic stats
    edges = mask.sum()
    print(f"  Edges: {edges}  Density: {edges/(H*H)*100:.2f}%")
    print(f"  In-degree:  mean={in_deg.mean():.1f} std={in_deg.std():.1f} [{in_deg.min()}-{in_deg.max()}]")
    print(f"  Out-degree: mean={out_deg.mean():.1f} std={out_deg.std():.1f} [{out_deg.min()}-{out_deg.max()}]")

    # Hub neurons (top 10 by total degree)
    hub_idx = np.argsort(total_deg)[::-1][:10]
    print(f"\n  TOP 10 HUBS (by total degree):")
    print(f"  {'neuron':>7s} {'in':>4s} {'out':>4s} {'tot':>4s} {'pol':>5s} {'theta':>5s} {'ch':>3s} {'zone':>8s}")
    for i in hub_idx:
        zone = 'INPUT' if i < IN_DIM else ('OUTPUT' if i >= H-OUT_DIM else 'HIDDEN')
        if i < IN_DIM and i >= H-OUT_DIM: zone = 'OVERLAP'
        pol = 'exc' if polarity[i] else 'INH'
        print(f"  {i:7d} {in_deg[i]:4d} {out_deg[i]:4d} {total_deg[i]:4d} {pol:>5s} {theta[i]:5d} {channel[i]:3d} {zone:>8s}")

    # Reciprocal connections (A→B AND B→A)
    recip = (mask & mask.T).sum() // 2
    print(f"\n  Reciprocal pairs: {recip} ({recip*2/max(edges,1)*100:.1f}% of edges)")

    # Zone connectivity matrix
    overlap_start = H - OUT_DIM
    zones = np.zeros(H, dtype=int)  # 0=input-only, 1=overlap, 2=output-only
    for i in range(H):
        if i < IN_DIM and i >= overlap_start:
            zones[i] = 1  # overlap
        elif i < IN_DIM:
            zones[i] = 0  # input only
        else:
            zones[i] = 2  # output only
    zone_names = ['IN-only', 'OVERLAP', 'OUT-only']
    print(f"\n  ZONE CONNECTIVITY (edges between zones):")
    print(f"  Zone sizes: {[(zones==z).sum() for z in range(3)]}")
    header = 'from\\to'
    print(f"  {header:>10s}", end="")
    for z in range(3): print(f"  {zone_names[z]:>10s}", end="")
    print()
    for zf in range(3):
        print(f"  {zone_names[zf]:>10s}", end="")
        for zt in range(3):
            src_mask = zones == zf
            tgt_mask = zones == zt
            count = mask[np.ix_(src_mask, tgt_mask)].sum()
            print(f"  {count:10d}", end="")
        print()

    # Inhibitory hub analysis
    inh_idx = np.where(~polarity)[0]
    exc_idx = np.where(polarity)[0]
    if len(inh_idx) > 0:
        inh_out = out_deg[inh_idx]
        exc_out = out_deg[exc_idx]
        print(f"\n  INHIBITORY ANALYSIS:")
        print(f"  Inh neurons: {len(inh_idx)} ({len(inh_idx)/H*100:.1f}%)")
        print(f"  Inh out-degree: mean={inh_out.mean():.1f} [{inh_out.min()}-{inh_out.max()}]")
        print(f"  Exc out-degree: mean={exc_out.mean():.1f} [{exc_out.min()}-{exc_out.max()}]")
        print(f"  Inh/Exc fan-out ratio: {inh_out.mean()/max(exc_out.mean(),0.01):.2f}x")

    # Theta by zone
    print(f"\n  THETA BY ZONE:")
    for z in range(3):
        zmask = zones == z
        if zmask.sum() > 0:
            th = theta[zmask].astype(float)
            print(f"  {zone_names[z]:>10s}: mean={th.mean():.1f} std={th.std():.1f}")

    # Channel distribution
    ch_counts = np.bincount(channel, minlength=9)[1:9]
    print(f"\n  CHANNEL DISTRIBUTION:")
    for c in range(8):
        bar = '#' * int(ch_counts[c] / max(ch_counts.max(), 1) * 20)
        print(f"  ch{c+1}: {ch_counts[c]:3d} |{bar}")

    # Degree distribution ASCII histogram
    print(f"\n  IN-DEGREE HISTOGRAM:")
    hist, edges_h = np.histogram(in_deg, bins=10)
    for i in range(10):
        bar = '#' * int(hist[i] / max(hist.max(), 1) * 30)
        print(f"  [{edges_h[i]:5.1f}-{edges_h[i+1]:5.1f}) {hist[i]:3d} |{bar}")

    sys.stdout.flush()
    return {
        'edges': int(edges), 'recip': int(recip),
        'in_deg_mean': float(in_deg.mean()), 'in_deg_std': float(in_deg.std()),
        'out_deg_mean': float(out_deg.mean()),
        'hub_top5': [(int(i), int(total_deg[i])) for i in hub_idx[:5]],
    }

def compare_snapshots(mask_old, mask_new, theta_old, theta_new, label=""):
    """Compare two snapshots: what changed?"""
    added = mask_new & ~mask_old   # new edges
    removed = mask_old & ~mask_new  # removed edges
    n_added = added.sum()
    n_removed = removed.sum()
    n_same = (mask_old & mask_new).sum()

    # Theta changes
    th_changed = (theta_old != theta_new).sum()

    print(f"\n  DIFF: {label}")
    print(f"  Edges: +{n_added} -{n_removed} ={n_same} (turnover: {(n_added+n_removed)/max(n_same,1)*100:.1f}%)")
    print(f"  Theta changed: {th_changed}/{H} neurons")

    # Where do new edges go?
    if n_added > 0:
        add_rows, add_cols = np.where(added)
        # Source zone distribution
        in_src = (add_rows < IN_DIM).sum()
        out_src = (add_rows >= H-OUT_DIM).sum()
        # Target zone distribution
        in_tgt = (add_cols < IN_DIM).sum()
        out_tgt = (add_cols >= H-OUT_DIM).sum()
        print(f"  New edges: from IN={in_src} from OUT={out_src} | to IN={in_tgt} to OUT={out_tgt}")

    sys.stdout.flush()

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    pol_bool = ref.polarity.copy()
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    # Init snapshot
    analyze_structure(mask, pol_bool, theta.astype(np.uint8), channel, "INIT (step 0)")
    snapshots = [{'step': 0, 'mask': mask.copy(), 'theta': theta.astype(np.uint8).copy()}]

    init_w(bp_out, ALL_DATA, bigram, pol_f32, channel)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, channel))

    best=0;acc=0;t0=time.time()
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in('flip','remove','theta','channel') and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),channel.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1
            elif br['type']=='channel' and br['new_channel'] is not None:
                channel[:]=br['new_channel'];acc+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            for s in eval_seqs:
                sc=SelfWiringGraph.build_sparse_cache(mask)
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                        decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                        state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,
                        channel=channel)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                print(f"\n  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} edges={mask.sum()} {time.time()-t0:.0f}s")
                sys.stdout.flush()

        # Snapshot every 500 steps
        if step in (500, 1000, 1500, 2000):
            snap = {'step': step, 'mask': mask.copy(), 'theta': theta.astype(np.uint8).copy()}
            analyze_structure(mask, pol_bool, theta.astype(np.uint8), channel, f"STEP {step} (best={best*100:.1f}%)")
            compare_snapshots(snapshots[-1]['mask'], mask,
                            snapshots[-1]['theta'], theta.astype(np.uint8),
                            f"step {snapshots[-1]['step']} -> {step}")
            snapshots.append(snap)

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    # Final: compare init to end
    print(f"\n\n{'='*60}")
    print(f"  OVERALL EVOLUTION: init -> step 2000")
    print(f"{'='*60}")
    compare_snapshots(snapshots[0]['mask'], snapshots[-1]['mask'],
                     snapshots[0]['theta'], snapshots[-1]['theta'],
                     "INIT -> FINAL")

    # Edge survival: which init edges survived to the end?
    init_edges = snapshots[0]['mask']
    final_edges = snapshots[-1]['mask']
    survived = (init_edges & final_edges).sum()
    born = (final_edges & ~init_edges).sum()
    died = (init_edges & ~final_edges).sum()
    print(f"\n  Edge lifecycle:")
    print(f"  Survived from init: {survived} ({survived/max(init_edges.sum(),1)*100:.1f}%)")
    print(f"  Born during training: {born}")
    print(f"  Died during training: {died}")
    print(f"  Turnover: {(born+died)/(survived+born)*100:.1f}%")

    print(f"\n  TOTAL TIME: {elapsed:.0f}s")
    print(f"  FINAL BEST: {best*100:.1f}%")
    sys.stdout.flush()
