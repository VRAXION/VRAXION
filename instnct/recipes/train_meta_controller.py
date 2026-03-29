"""
Long-run meta controller training
===================================
Train a H=32 mini INSTNCT controller for 10K steps or until plateau.
Bigger than H=16, enough capacity to learn complex strategies.
Checkpoint every 1K steps. Plateau = 2K steps without improvement.
Goal: the best possible mutation scheduling network.
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
WAVE_LUT = SelfWiringGraph.WAVE_LUT

ALL_OPS = ['add','flip','theta','channel','reverse','remove']
N_OPS = len(ALL_OPS)

# Controller: bigger this time
CH = 32
C_IN = 9   # 6 tool rates + accuracy + progress + last_delta_magnitude
C_OUT = 6

class MetaController:
    """H=32 mini INSTNCT for mutation scheduling. Evolves itself."""
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        self.H = CH
        self.mask = (rng.rand(CH,CH) < 0.20).astype(bool)
        np.fill_diagonal(self.mask, False)
        self.theta = rng.randint(1, 8, size=CH).astype(np.uint8)
        self.channel = rng.randint(1, 9, size=CH).astype(np.uint8)
        self.polarity = rng.rand(CH) > 0.2
        self.pol_f32 = np.where(self.polarity, 1.0, -1.0).astype(np.float32)
        self.recent_picks = []
        self.fitness = 0.5

    def vote(self, tool_rates, accuracy, progress, last_delta):
        inj = np.zeros(CH, np.float32)
        for i, op in enumerate(ALL_OPS):
            inj[i] = tool_rates.get(op, 0.5) * 10.0
        inj[6] = accuracy * 10.0
        inj[7] = progress * 10.0
        inj[8] = min(abs(last_delta) * 1000, 10.0)  # magnitude signal

        sc = SelfWiringGraph.build_sparse_cache(self.mask)
        state = np.zeros(CH, np.float32)
        charge = np.zeros(CH, np.float32)
        spike_counts = np.zeros(CH, np.float32)
        theta_f = self.theta.astype(np.float32)

        for tick in range(TICKS):
            if tick % 6 == 0:
                charge = np.maximum(charge - 1.0, 0.0)
            if tick < 2:
                state = state + inj
            if len(sc) >= 2 and len(sc[0]) > 0:
                raw = np.zeros(CH, np.float32)
                np.add.at(raw, sc[1], state[sc[0]])
                charge += raw
            np.clip(charge, 0.0, 15.0, out=charge)
            theta_mult = WAVE_LUT[self.channel, tick % 8]
            eff_theta = np.clip(theta_f * theta_mult, 1.0, 15.0)
            fired = charge >= eff_theta
            spike_counts += fired.astype(np.float32)
            state = fired.astype(np.float32) * self.pol_f32
            charge[fired] = 0.0

        return spike_counts[CH-C_OUT:]

    def save_state(self):
        return {'mask': self.mask.copy(), 'theta': self.theta.copy(),
                'channel': self.channel.copy(), 'polarity': self.polarity.copy(),
                'pol_f32': self.pol_f32.copy()}

    def restore_state(self, s):
        self.mask[:]=s['mask']; self.theta[:]=s['theta']; self.channel[:]=s['channel']
        self.polarity[:]=s['polarity']; self.pol_f32[:]=s['pol_f32']

    def mutate(self):
        op = random.choice(['add','remove','theta','channel','flip_pol'])
        if op == 'add':
            r=random.randint(0,CH-1);c=random.randint(0,CH-1)
            if r!=c and not self.mask[r,c]: self.mask[r,c]=True
        elif op == 'remove':
            alive=list(zip(*np.where(self.mask)))
            if alive: r,c=random.choice(alive); self.mask[r,c]=False
        elif op == 'theta':
            self.theta[random.randint(0,CH-1)]=np.uint8(random.randint(1,8))
        elif op == 'channel':
            self.channel[random.randint(0,CH-1)]=np.uint8(random.randint(1,8))
        elif op == 'flip_pol':
            idx=random.randint(0,CH-1)
            self.polarity[idx]=not self.polarity[idx]
            self.pol_f32[idx]=1.0 if self.polarity[idx] else -1.0

    def update_fitness(self, accepted):
        self.recent_picks.append(1.0 if accepted else 0.0)
        if len(self.recent_picks) > 100:
            self.recent_picks = self.recent_picks[-100:]
        self.fitness = np.mean(self.recent_picks)

    def save_checkpoint(self, path):
        np.savez_compressed(path, mask=self.mask, theta=self.theta,
            channel=self.channel, polarity=self.polarity, pol_f32=self.pol_f32)

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

    MAX_STEPS = 10000
    PLATEAU_LIMIT = 2000
    CTRL_AB_EVERY = 100  # A/B test controller every 100 steps
    CTRL_AB_HALF = 50    # 50 steps per half (new vs old)
    CKPT_EVERY = 1000

    print(f"\n{'='*60}")
    print(f"  META CONTROLLER LONG TRAINING")
    print(f"  Controller: H={CH}, max {MAX_STEPS} steps, plateau={PLATEAU_LIMIT}")
    print(f"  Main: H={H} (fresh each run, controller learns across)")
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

    ctrl = MetaController(seed=77)
    tool_rates = {op: 0.5 for op in ALL_OPS}
    op_history = {op: [0.5]*5 for op in ALL_OPS}
    current_accuracy = 0.0
    last_delta = 0.0

    init_w(bp_out, ALL_DATA, bigram, pol_f32)
    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;t0=time.time();stall=0;step=0
    op_counts = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}
    ctrl_mutations=0;ctrl_improvements=0;ctrl_reverts=0

    # Helper: run N steps, return accepts count (modifies mask/theta/etc in place)
    def do_steps(n_steps, ctrl_ref, m, th, ch, pf):
        """Run n steps, return (accepts, final_m, final_th, final_ch, final_pf)."""
        local_acc = 0
        for _ in range(n_steps):
            votes = ctrl_ref.vote(tool_rates, current_accuracy, step/MAX_STEPS, last_delta)
            if votes.sum() > 0:
                lg = votes / 1.5; lg -= lg.max()
                pr = np.exp(lg) / (np.exp(lg).sum() + 1e-8)
            else:
                pr = np.ones(N_OPS) / N_OPS
            if random.random() < 0.08:
                pt = random.choice(ALL_OPS)
            else:
                pt = ALL_OPS[np.random.choice(N_OPS, p=pr)]
            if m.sum()==0: pt='add'

            args=[(m.flatten(),th.copy(),ch.copy(),pf.copy(),
                   1000+(step+_)*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta'] > THRESHOLD:
                if br['new_mask_flat'] is not None: m[:]=br['new_mask_flat'].reshape(H,H)
                if br['new_theta'] is not None: th[:]=br['new_theta']
                if br['new_channel'] is not None: ch[:]=br['new_channel']
                if br['new_pol_f'] is not None: pf[:]=br['new_pol_f']
                local_acc += 1
        return local_acc, m, th, ch, pf

    # Main loop
    while step < MAX_STEPS:
        # Save state
        saved_ctrl = ctrl.save_state()
        m_save = mask.copy(); th_save = theta.copy()
        ch_save = channel.copy(); pf_save = pol_f32.copy()

        # Phase A: 50 steps with CURRENT controller
        m_a = m_save.copy(); th_a = th_save.copy()
        ch_a = ch_save.copy(); pf_a = pf_save.copy()
        old_accepts, m_a, th_a, ch_a, pf_a = do_steps(CTRL_AB_HALF, ctrl, m_a, th_a, ch_a, pf_a)

        # Phase B: mutate controller, 50 steps with NEW
        ctrl.mutate()
        ctrl_mutations += 1
        m_b = m_save.copy(); th_b = th_save.copy()
        ch_b = ch_save.copy(); pf_b = pf_save.copy()
        new_accepts, m_b, th_b, ch_b, pf_b = do_steps(CTRL_AB_HALF, ctrl, m_b, th_b, ch_b, pf_b)

        step += CTRL_AB_HALF  # only count once (both ran same 50 steps worth)

        # Decision
        if new_accepts > old_accepts:
            ctrl_improvements += 1; verdict = "KEEP"
            mask[:]=m_b; theta[:]=th_b; channel[:]=ch_b; pol_f32[:]=pf_b
        elif new_accepts == old_accepts:
            if random.random() < 0.5:
                ctrl_improvements += 1; verdict = "KEEP(tie)"
                mask[:]=m_b; theta[:]=th_b; channel[:]=ch_b; pol_f32[:]=pf_b
            else:
                ctrl.restore_state(saved_ctrl); ctrl_reverts += 1; verdict = "REVERT(tie)"
                mask[:]=m_a; theta[:]=th_a; channel[:]=ch_a; pol_f32[:]=pf_a
        else:
            ctrl.restore_state(saved_ctrl); ctrl_reverts += 1; verdict = "REVERT"
            mask[:]=m_a; theta[:]=th_a; channel[:]=ch_a; pol_f32[:]=pf_a

        acc += max(old_accepts, new_accepts)

        # Eval
        ea_list=[]
        for s in eval_seqs:
            sc=SelfWiringGraph.build_sparse_cache(mask)
            st=np.zeros(H,np.float32);ch_e=np.zeros(H,np.float32);cor=0;tot=0
            for i in range(len(s)-1):
                inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                st,ch_e=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                    decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                    state=st,charge=ch_e,sparse_cache=sc,polarity=pol_f32,channel=channel)
                logits_e=np.dot(bp_out,ch_e[H-OUT_DIM:])
                if np.argmax(logits_e)==s[i+1]:cor+=1
                tot+=1
            ea_list.append(cor/tot if tot else 0)
        ea=np.mean(ea_list)
        current_accuracy = ea
        if ea>best: best=ea;stall=0
        else: stall+=CTRL_AB_HALF

        # Update tool rates from winner
        for op in ALL_OPS:
            op_history[op].append(tool_rates.get(op, 0.5))
            if len(op_history[op]) > 100: op_history[op] = op_history[op][-100:]
            tool_rates[op] = np.mean(op_history[op])

        # Report
        votes = ctrl.vote(tool_rates, current_accuracy, step/MAX_STEPS, last_delta)
        if votes.sum() > 0:
            lg = votes / 1.5; lg -= lg.max()
            probs = np.exp(lg) / (np.exp(lg).sum() + 1e-8)
        else:
            probs = np.ones(N_OPS) / N_OPS
        v_str=' '.join(f"{ALL_OPS[i]}={votes[i]:.0f}" for i in range(N_OPS))
        p_str=' '.join(f"{ALL_OPS[i]}={probs[i]:.0%}" for i in range(N_OPS))
        print(f"  [{step:5d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} stall={stall}")
        print(f"    Spikes: {v_str}")
        print(f"    Probs: {p_str}")
        print(f"    AB: old={old_accepts} new={new_accepts} -> {verdict}")
        print(f"    Ctrl: fit={ctrl.fitness:.0%} edges={ctrl.mask.sum()} mut={ctrl_mutations} imp={ctrl_improvements} rev={ctrl_reverts}")
        print(f"    {time.time()-t0:.0f}s")
        sys.stdout.flush()

        # Checkpoint
        if step % CKPT_EVERY < CTRL_AB_HALF + 1:
            ckpt_path = os.path.join(BASE_DIR, f"data/meta_ctrl_step{step}.npz")
            ctrl.save_checkpoint(ckpt_path)
            print(f"    CHECKPOINT: {ckpt_path}")

        # Plateau
        if stall >= PLATEAU_LIMIT and step >= 3000:
            print(f"    PLATEAU at step {step} (stall={stall})")
            break

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    # Save final
    final_path = os.path.join(BASE_DIR, "data/meta_ctrl_FINAL.npz")
    ctrl.save_checkpoint(final_path)

    print(f"\n{'='*60}")
    print(f"  META CONTROLLER FINAL")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% steps={step} {elapsed:.0f}s")
    print(f"  Controller: H={CH} fit={ctrl.fitness:.0%} edges={ctrl.mask.sum()}")
    print(f"  Mutations: {ctrl_mutations}, improvements: {ctrl_improvements}")
    print(f"  Ops: {dict(op_counts)}")
    print(f"  Accepts: {dict(op_accepts)}")
    print(f"  Saved: {final_path}")
