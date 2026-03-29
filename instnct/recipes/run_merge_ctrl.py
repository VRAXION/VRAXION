"""
MERGE: controller beolvad a main hálózatba
============================================
1. Load main checkpoint (H=256) + controller checkpoint (H=16)
2. Merge into H=272: main [0:256] + controller [256:272]
3. Remove isolation wall: mutations can create cross-edges
4. Continue training — see if the networks grow together

The controller's learned strategy (flip->theta->reverse phasing)
becomes an embedded sub-circuit in the main network.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
PHI = (1 + 5**0.5) / 2

ALL_OPS = ['add','flip','theta','channel','reverse','remove']
N_OPS = len(ALL_OPS)
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']

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

def merge_networks(main_ckpt_path, ctrl_ckpt_path):
    """Merge main (256) + controller (16) into one H=272 network."""
    main = np.load(main_ckpt_path)
    ctrl = np.load(ctrl_ckpt_path)

    H_main = main['mask'].shape[0]
    H_ctrl = ctrl['mask'].shape[0]
    H_total = H_main + H_ctrl

    # Merge mask: block diagonal (no cross-edges yet — evolution will add them)
    mask = np.zeros((H_total, H_total), dtype=bool)
    mask[:H_main, :H_main] = main['mask']
    mask[H_main:, H_main:] = ctrl['mask']

    # Merge theta
    theta = np.ones(H_total, dtype=np.float32)
    theta[:H_main] = main['theta']
    theta[H_main:] = ctrl['theta'].astype(np.float32)

    # Merge channel
    channel = np.ones(H_total, dtype=np.uint8)
    channel[:H_main] = main.get('channel', np.random.randint(1, 9, H_main).astype(np.uint8))
    channel[H_main:] = ctrl['channel']

    # Merge polarity
    pol_f32 = np.ones(H_total, dtype=np.float32)
    pol_f32[:H_main] = main['pol_f32']
    pol_f32[H_main:] = ctrl['pol_f32']

    print(f"  Merged: H_main={H_main} + H_ctrl={H_ctrl} = H_total={H_total}")
    print(f"  Main edges: {main['mask'].sum()}, Ctrl edges: {ctrl['mask'].sum()}")
    print(f"  Cross edges: 0 (isolation wall — evolution will add)")
    print(f"  Total edges: {mask.sum()}")

    return mask, theta, channel, pol_f32, H_total, H_main, H_ctrl

_bp_out=None;_all_data=None;_bigram=None

def init_w(bpo,data,bg):
    global _bp_out,_all_data,_bigram
    _bp_out=bpo;_all_data=data;_bigram=bg

def _eval_bigram(mask, theta, channel, pol_f32, H, IN_DIM, OUT_DIM, BP_IN, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    WAVE_LUT = SelfWiringGraph.WAVE_LUT
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=pol_f32,
                channel=channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:H])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,channel,pol_f,H,IN_DIM,OUT_DIM,bp_in_flat,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    BP_IN=bp_in_flat.reshape(256,IN_DIM)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel;npf=pol_f
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='flip':
        idx=rng.randint(0,H-1)
        pb=(pol_f>0);pb[idx]=not pb[idx]
        npf=np.where(pb,1.0,-1.0).astype(np.float32)
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
    elif pt=='reverse':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'reverse'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if r==c or mask[c,r]:return{'delta':-1e9,'type':'reverse'}
        nm=mask.copy();nm[r,c]=False;nm[c,r]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,channel,pol_f,H,IN_DIM,OUT_DIM,BP_IN,seqs)
    new=_eval_bigram(nm,nt,nc,npf,H,IN_DIM,OUT_DIM,BP_IN,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_f':npf if pt=='flip' else None}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))

    # Check if checkpoints exist
    main_ckpt = os.path.join(BASE_DIR, "data", "mini_ctrl_main_network.npz")
    ctrl_ckpt = os.path.join(BASE_DIR, "data", "mini_ctrl_controller.npz")

    if not os.path.exists(main_ckpt) or not os.path.exists(ctrl_ckpt):
        print(f"  WAITING: checkpoints not ready yet")
        print(f"    main: {os.path.exists(main_ckpt)}")
        print(f"    ctrl: {os.path.exists(ctrl_ckpt)}")
        print(f"  Run run_mini_instnct_ctrl.py first!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  MERGE: main (256) + controller (16) = 272 neurons")
    print(f"{'='*60}")

    mask, theta, channel, pol_f32, H, H_main, H_ctrl = merge_networks(main_ckpt, ctrl_ckpt)

    # Recalculate I/O dims for merged network
    IN_DIM = int(round(H_main / PHI))  # keep original input zone size
    OUT_DIM = int(round(H_main / PHI))  # keep original output zone size
    SDR_K = int(round(IN_DIM * 0.20))
    BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)
    bp_out = build_freq_order(OUT_DIM, bigram)

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"  H={H} IN_DIM={IN_DIM} OUT_DIM={OUT_DIM}")
    print(f"  Main zone: [0:{H_main}], Controller zone: [{H_main}:{H}]")

    # Pre-merge eval
    init_w(bp_out, ALL_DATA, bigram)

    print(f"\n  Pre-merge eval...")
    ea_list=[]
    for s in eval_seqs:
        sc=SelfWiringGraph.build_sparse_cache(mask)
        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
        for i in range(len(s)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
            logits=np.dot(bp_out,ch[H-OUT_DIM:H])
            if np.argmax(logits)==s[i+1]:cor+=1
            tot+=1
        ea_list.append(cor/tot if tot else 0)
    pre_merge_acc = np.mean(ea_list)
    print(f"  Pre-merge accuracy: {pre_merge_acc*100:.1f}%")
    sys.stdout.flush()

    # Train merged network
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram))

    best=pre_merge_acc;acc=0;t0=time.time()
    op_counts = {op: 0 for op in ALL_OPS}

    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in SCHEDULE and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               H,IN_DIM,OUT_DIM,BP_IN.flatten(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None:theta[:]=br['new_theta']
            if br['new_channel'] is not None:channel[:]=br['new_channel']
            if br['new_pol_f'] is not None:pol_f32[:]=br['new_pol_f']
            acc+=1;op_counts[br['type']]+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            for s in eval_seqs:
                sc=SelfWiringGraph.build_sparse_cache(mask)
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                        decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                        state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:H])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                # Count cross-edges (main<->ctrl)
                cross = mask[:H_main, H_main:].sum() + mask[H_main:, :H_main].sum()
                ctrl_edges = mask[H_main:, H_main:].sum()
                main_edges = mask[:H_main, :H_main].sum()
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f}")
                print(f"    Edges: main={main_edges} ctrl={ctrl_edges} CROSS={cross} total={mask.sum()}")
                print(f"    Ops: {dict(op_counts)}")
                print(f"    {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    cross = mask[:H_main, H_main:].sum() + mask[H_main:, :H_main].sum()
    print(f"\n{'='*60}")
    print(f"  MERGE RESULTS (H={H})")
    print(f"{'='*60}")
    print(f"  Pre-merge: {pre_merge_acc*100:.1f}%")
    print(f"  Post-merge best: {best*100:.1f}%")
    print(f"  Cross edges: {cross} (bridges between main and controller)")
    print(f"  Ops: {dict(op_counts)}")
    print(f"  {elapsed:.0f}s")
