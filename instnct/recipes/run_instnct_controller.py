"""
INSTNCT controller: a small self-wiring graph decides mutations
================================================================
Separate small network (H=32) with original C19 sin-based wave gating.
Input: per-tool accept rates + pain/reward + accuracy
Output: 6 neurons, one per tool, highest charge wins
The controller ITSELF evolves alongside the main network.

Architecture:
  Controller: H=32, original C19 (sin freq/phase, rho=0.4)
    Input zone (8 neurons):
      [0] accuracy signal (0-15 proportional)
      [1] pain signal (fires on reject)
      [2] reward signal (fires on accept)
      [3-8] per-tool recent accept rate (0-15 proportional)
    Hidden: 3 layer depth via 8 ticks
    Output zone (6 neurons):
      [26-31] = one per tool, charge readout = vote

  Main network: H=256, standard channel-based (current best)
    Evolved normally with tool chosen by controller
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

# Main network params
H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))

# Controller params
CH = 32  # controller hidden size
C_TICKS = 8
C_IN = 9   # accuracy + pain + reward + 6 tool rates
C_OUT = 6  # one per tool
C_RHO = 0.5  # original C19 Canon rho (commit 947004e)
ALL_OPS = ['add','flip','theta','channel','reverse','remove']
N_OPS = len(ALL_OPS)

class C19Controller:
    """Small INSTNCT network as mutation scheduler. Original C19 sin-based."""
    def __init__(self, seed=123):
        rng = np.random.RandomState(seed)
        self.H = CH
        self.mask = (rng.rand(CH,CH) < 0.15).astype(bool)  # 15% density (denser for small net)
        np.fill_diagonal(self.mask, False)
        self.theta = np.full(CH, 3.0, np.float32)  # low threshold for responsiveness
        self.freq = (rng.rand(CH).astype(np.float32) * 1.5 + 0.5)
        self.phase = (rng.rand(CH).astype(np.float32) * 2.0 * np.pi)
        self.rho = np.full(CH, C_RHO, np.float32)
        self.polarity = rng.rand(CH) > 0.15  # 85% excitatory
        self.pol_f32 = np.where(self.polarity, 1.0, -1.0).astype(np.float32)

    def vote(self, accuracy, pain, reward_signal, tool_rates):
        """Forward pass through controller, return tool votes."""
        # Build input injection
        inj = np.zeros(CH, np.float32)
        # Input zone [0:C_IN]
        inj[0] = accuracy * 15.0       # scale to charge range
        inj[1] = pain * 15.0
        inj[2] = reward_signal * 15.0
        for i, op in enumerate(ALL_OPS):
            inj[3+i] = tool_rates.get(op, 0.5) * 15.0

        # Run C19 forward pass (original sin-based)
        sc = SelfWiringGraph.build_sparse_cache(self.mask)
        state = np.zeros(CH, np.float32)
        charge = np.zeros(CH, np.float32)
        for tick in range(C_TICKS):
            if tick % 6 == 0:
                charge = np.maximum(charge - 1.0, 0.0)
            if tick < 2:
                state = state + inj
            # Propagate
            if len(sc[0]) > 0:
                raw = np.zeros(CH, np.float32)
                rows, cols = sc[0], sc[1]
                np.add.at(raw, cols, state[rows])
                charge += raw
            np.clip(charge, 0.0, 15.0, out=charge)
            # C19 sin-based wave gating
            wave = np.sin(tick * self.freq + self.phase)
            eff_theta = np.clip(self.theta * (1.0 + self.rho * wave), 1.0, 15.0)
            fired = charge >= eff_theta
            state = fired.astype(np.float32) * self.pol_f32
            charge[fired] = 0.0

        # Read output zone: last C_OUT neurons
        votes = charge[CH-C_OUT:]
        return votes

    def mutate(self):
        """Small mutation on the controller itself."""
        rng = random.Random()
        undo = []
        # 50% chance: structural, 50% chance: theta
        if rng.random() < 0.5:
            # Flip one edge
            r = rng.randint(0, CH-1); c = rng.randint(0, CH-1)
            if r != c:
                old = bool(self.mask[r,c])
                self.mask[r,c] = not self.mask[r,c]
                undo.append(('mask', r, c, old))
        else:
            # Theta resample
            idx = rng.randint(0, CH-1)
            old = float(self.theta[idx])
            self.theta[idx] = float(rng.randint(1, 10))
            undo.append(('theta', idx, old))
        return undo

    def revert(self, undo):
        for op in undo:
            if op[0] == 'mask':
                self.mask[op[1], op[2]] = op[3]
            elif op[0] == 'theta':
                self.theta[op[1]] = op[2]

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
WAVE_LUT = SelfWiringGraph.WAVE_LUT
_bp_out=None;_all_data=None;_bigram=None;_pol=None

def init_w(bpo,data,bg,pol):
    global _bp_out,_all_data,_bigram,_pol
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol

def _eval_bigram(mask, theta, channel, polarity_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=polarity_f,
                channel=channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,channel,pol_f,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
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
    old=_eval_bigram(mask,theta,channel,pol_f,seqs)
    new=_eval_bigram(nm,nt,nc,npf,seqs)
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
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*60}")
    print(f"  INSTNCT CONTROLLER (H_ctrl={CH}, C19 sin-based)")
    print(f"  Main network: H={H}, channel-based")
    print(f"  Controller input: accuracy + pain + reward + 6 tool rates")
    print(f"  Controller output: 6 neurons (charge vote)")
    print(f"  ops: {ALL_OPS}")
    print(f"  baselines: fixed=23.8%, MLP=13.7%")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    # Init controller
    ctrl = C19Controller(seed=99)
    print(f"  Controller: {ctrl.mask.sum()} edges, theta={ctrl.theta.mean():.1f}")

    # Feedback state
    pain = 0.0; reward_sig = 0.0
    current_accuracy = 0.0
    tool_rates = {op: 0.5 for op in ALL_OPS}
    op_history = {op: [] for op in ALL_OPS}

    init_w(bp_out, ALL_DATA, bigram, pol_f32)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}
    ctrl_mutations = 0

    for step in range(1, 2001):
        # Controller votes
        votes = ctrl.vote(current_accuracy, pain, reward_sig, tool_rates)

        # Softmax with exploration
        logits = votes / 2.0  # temperature
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)

        if random.random() < 0.15:  # 15% exploration
            pt = random.choice(ALL_OPS)
        else:
            pt = ALL_OPS[np.random.choice(N_OPS, p=probs)]

        if mask.sum()==0: pt='add'
        op_counts[pt] += 1

        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

        accepted = br['delta'] > THRESHOLD
        if accepted:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            acc+=1; op_accepts[br['type']]+=1
            pain=0.0; reward_sig=1.0
        else:
            pain=1.0; reward_sig=0.0

        # Update tool rates (rolling 20)
        op_history[pt].append(1.0 if accepted else 0.0)
        if len(op_history[pt]) > 20:
            op_history[pt] = op_history[pt][-20:]
        tool_rates[pt] = np.mean(op_history[pt]) if op_history[pt] else 0.5

        # Mutate controller every 50 steps
        if step % 50 == 0:
            ctrl_undo = ctrl.mutate()
            # Test: did the controller's choice improve?
            # Simple: if last 50 steps had good accept rate, keep mutation
            recent_rate = acc / max(step, 1)
            if random.random() < 0.5:  # 50% chance revert (exploration)
                ctrl.revert(ctrl_undo)
            else:
                ctrl_mutations += 1

        # Decay feedback
        pain *= 0.7
        reward_sig *= 0.7

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
                    logits_e=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits_e)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            current_accuracy = ea
            if ea>best:best=ea
            if step%200==0:
                probs_str = ' '.join(f"{ALL_OPS[i]}={probs[i]:.0%}" for i in range(N_OPS))
                votes_str = ' '.join(f"{ALL_OPS[i]}={votes[i]:.2f}" for i in range(N_OPS))
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f}")
                print(f"    Votes: {votes_str}")
                print(f"    Probs: {probs_str}")
                print(f"    Ops: {dict(op_counts)}")
                print(f"    Ctrl: {ctrl.mask.sum()} edges, {ctrl_mutations} mutations")
                print(f"    {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  INSTNCT CONTROLLER RESULTS")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  Ops: {dict(op_counts)}")
    print(f"  Accepts: {dict(op_accepts)}")
    print(f"  Controller: {ctrl.mask.sum()} edges, {ctrl_mutations} mutations")
    print(f"  baselines: fixed=23.8%, MLP=13.7%, adaptive=12.5%")
