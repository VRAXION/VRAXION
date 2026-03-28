"""
Tree-structured pre-wiring: 4-way frequency-balanced tournament tree
====================================================================
The mask starts with a tree scaffold:
  Level 0: ROOT neuron (receives from all input, sends to 4 group neurons)
  Level 1: 4 group neurons (send to 4 subgroups each)
  Level 2: 16 subgroup neurons (send to 4 leaf groups each)
  Level 3: 64 leaf neurons (map to final bytes)

Plus: top-20 bigram shortcut edges (direct input→output for common pairs)
Evolution then sculpts: prune bad branches, add new shortcuts, flip edges.

H=256, phi overlap, SDR input, FREQ output, learnable theta.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20; THETA_INIT = 1.0
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))   # 158
OUT_DIM = int(round(H / PHI))  # 158
SDR_K = int(round(IN_DIM * 0.20))

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            freq_d = (d + 1) / dim
            p[byte_idx, d] = np.sin(2 * np.pi * t * freq_d * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

def build_tree_mask(bigram):
    """Build 4-way frequency-balanced tree scaffold in the mask."""
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    mask = np.zeros((H, H), dtype=bool)

    # Neuron allocation: input=[0,157], output=[98,255]
    # Tree neurons live in the overlap zone [98,157] = 60 neurons
    # We need: 1 root + 4 L1 + 16 L2 = 21 tree neurons (fits easily)
    # Place tree neurons at the START of overlap zone
    tree_start = 98  # overlap start
    ROOT_N = tree_start
    L1 = [tree_start + 1, tree_start + 2, tree_start + 3, tree_start + 4]
    L2 = [tree_start + 5 + i for i in range(16)]
    # L3 = the remaining output neurons (the actual byte readout neurons)

    # Split 256 bytes into 4 groups by frequency balance
    all_bytes = list(range(256))
    sorted_bytes = sorted(all_bytes, key=lambda b: freq[b], reverse=True)

    def split_4way(byte_list):
        """Split into 4 groups with balanced frequency."""
        groups = [[], [], [], []]
        sums = [0, 0, 0, 0]
        for b in sorted(byte_list, key=lambda b: freq[b], reverse=True):
            min_idx = sums.index(min(sums))
            groups[min_idx].append(b)
            sums[min_idx] += freq[b]
        return groups

    # Level 0→1: split all bytes into 4 groups
    groups_l1 = split_4way(sorted_bytes)

    # Level 1→2: split each L1 group into 4 subgroups
    groups_l2 = []
    for g in groups_l1:
        groups_l2.extend(split_4way(g))

    # Wire: input neurons → ROOT
    # SDR input activates neurons [0, IN_DIM). Connect a sample to ROOT.
    rng = np.random.RandomState(42)
    # Top-frequency input neurons → ROOT (not all, sparse connection)
    top_input = sorted(range(IN_DIM), key=lambda i: freq[i % 256] if i < 256 else 0, reverse=True)[:20]
    for inp in top_input:
        mask[inp, ROOT_N] = True

    # ROOT → L1 group neurons
    for g_neuron in L1:
        mask[ROOT_N, g_neuron] = True

    # L1 → L2 subgroup neurons
    for i, l1_neuron in enumerate(L1):
        for j in range(4):
            l2_idx = i * 4 + j
            if l2_idx < len(L2):
                mask[l1_neuron, L2[l2_idx]] = True

    # L2 → output neurons (map subgroup bytes to output zone)
    # Output zone = [H - OUT_DIM, H) = [98, 255]
    output_start = H - OUT_DIM
    for l2_idx, l2_neuron in enumerate(L2):
        if l2_idx < len(groups_l2):
            for byte_val in groups_l2[l2_idx]:
                # Map byte to output neuron position
                out_neuron = output_start + (byte_val % OUT_DIM)
                if out_neuron != l2_neuron:  # no self-edge
                    mask[l2_neuron, out_neuron] = True

    # Input neurons for each group → their L1 neuron
    for i, group_bytes in enumerate(groups_l1):
        for byte_val in group_bytes[:10]:  # top 10 bytes per group
            inp_neuron = byte_val % IN_DIM
            if inp_neuron != L1[i]:
                mask[inp_neuron, L1[i]] = True

    # Bigram shortcut edges: top-20 most common bigram pairs get direct input→output
    bigram_flat = []
    for a in range(256):
        for b in range(256):
            if bigram[a, b] > 0.05:  # > 5% probability
                bigram_flat.append((a, b, bigram[a, b]))
    bigram_flat.sort(key=lambda x: x[2], reverse=True)
    for a, b, prob in bigram_flat[:20]:
        inp_neuron = a % IN_DIM
        out_neuron = output_start + (b % OUT_DIM)
        if inp_neuron != out_neuron:
            mask[inp_neuron, out_neuron] = True

    np.fill_diagonal(mask, False)

    tree_edges = int(mask.sum())
    print(f"  Tree scaffold: {tree_edges} edges")
    print(f"  Tree neurons: ROOT={ROOT_N}, L1={L1}, L2={L2[:4]}...{L2[-1]}")
    print(f"  Groups L1: {[len(g) for g in groups_l1]} bytes")
    print(f"  Shortcuts: {min(20, len(bigram_flat))} bigram direct edges")

    return mask

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(bpo,data,bg,pol,fr,ph,rh):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nd=decay
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
        r,c=alive[rng.randint(0,len(alive)-1)];nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc]=True
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=rng.uniform(0.0,16.0)
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,nt,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_decay':nd if pt=='decay' else None}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    from lib.archive import save_experiment
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*60}")
    print(f"  TREE-WIRED SCAFFOLD + FREQ OUTPUT")
    print(f"  H={H}, phi overlap, 4-way tree")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)

    # Build tree scaffold (NOT random 5% — structured init)
    mask = build_tree_mask(bigram)
    # ALSO add random 3% on top for exploration material
    rng_init = np.random.RandomState(42)
    random_edges = (rng_init.rand(H, H) < 0.03).astype(bool)
    np.fill_diagonal(random_edges, False)
    mask = mask | random_edges
    np.fill_diagonal(mask, False)
    print(f"  Total after random padding: {int(mask.sum())} edges")

    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

    best=0;stall=0;acc=0;t_acc=0;add_a=0;rem_a=0;flip_a=0;dec_a=0;t0=time.time()
    try:
        for step in range(1, 2001):
            pt=SCHEDULE[(step-1)%len(SCHEDULE)]
            if pt in('flip','remove','decay','theta') and mask.sum()==0:pt='add'
            args=[(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                    if br['type']=='add':add_a+=1
                    elif br['type']=='remove':rem_a+=1
                    else:flip_a+=1
                elif br['type']=='theta' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];t_acc+=1;acc+=1
                elif br['type']=='decay' and br['new_decay'] is not None:
                    decay[:]=br['new_decay'];dec_a+=1;acc+=1
            if step%EVAL_EVERY==0:
                el=time.time()-t0;ea_list=[]
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                            sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea;stall=0
                else:stall+=EVAL_EVERY
                if step%100==0:
                    edges=int(mask.sum())
                    print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"edges={edges} theta={theta.mean():.2f} "
                          f"[A={add_a} R={rem_a} F={flip_a} T={t_acc} D={dec_a}] "
                          f"{el:.0f}s ({step/el:.1f}sps)")
                sys.stdout.flush()
                if stall>=800 and step>=600:print(f"  ** STALL");break
    finally:pool.terminate();pool.join()

    elapsed=time.time()-t0;edges=int(mask.sum())
    save_experiment(
        name="tree_wiring_freq_output",
        mask=mask, theta=theta, decay=decay, polarity=pol.astype(np.int8),
        bp_out=bp_out,
        config={'H':H,'in_dim':IN_DIM,'out_dim':OUT_DIM,'init':'tree_scaffold+3%random'},
        result={'best':round(best,4),'edges':edges,
                'add':add_a,'remove':rem_a,'flip':flip_a,'theta_acc':t_acc,'decay_acc':dec_a},
    )

    print(f"\n{'='*60}")
    print(f"  DONE: best={best*100:.1f}% edges={edges}")
    print(f"  [A={add_a} R={rem_a} F={flip_a} T={t_acc} D={dec_a}] total={acc}")
    print(f"  theta={theta.mean():.2f}[{theta.min():.1f},{theta.max():.1f}]")
    print(f"  Compare: random 5% init = 22.4%, no tree")
    print(f"{'='*60}")
