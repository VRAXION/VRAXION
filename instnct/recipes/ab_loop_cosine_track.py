"""
Loop vs Random vs Baseline — cosine distribution tracking
==========================================================
Track cosine score (distribution quality) not just accuracy.
This shows whether loops help the network approximate the
language distribution FASTER, even before argmax rank changes.

From crystallized checkpoint (3 edges, 21.64% = space-only).
30 edge injection budget, 1500 training steps.
Report cosine score every 100 steps.
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

CKPT = ROOT / 'recipes' / 'checkpoints' / 'variant_seed42_crystal.npz'
INJECTION_BUDGET = 30
TRAINING_STEPS = 1500
SEQ_LEN = 150
N_TRAIN = 2
N_EVAL = 8
REPORT_EVERY = 100
THRESHOLD = 0.00005
TICKS = 8
INPUT_DUR = 2
SEED = 42
SCHEDULE = ['add', 'add', 'flip', 'decay', 'decay', 'decay', 'decay', 'decay']

H = 1024; IO = 256

def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, IO).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def rollout(inj, mask, theta, decay, state, charge, sp, pol, ref):
    act=state.copy(); cur=charge.copy(); r=ref.copy()
    rows,cols=sp; df=np.asarray(decay,dtype=np.float32)
    is_sc=df.ndim==0 or df.shape==()
    dp=max(1,int(round(1.0/max(float(df),0.001)))) if is_sc else 0
    for tick in range(TICKS):
        if dp>0:
            if tick%dp==0: cur=np.maximum(cur-1.0,0.0)
        else: cur=np.maximum(cur-df,0.0)
        if tick<INPUT_DUR: act=act+inj
        raw=np.zeros(H,dtype=np.float32)
        if len(rows): np.add.at(raw,cols,act[rows])
        np.nan_to_num(raw,copy=False); cur+=raw; np.clip(cur,0.0,15.0,out=cur)
        can=(r==0); fired=(cur>=theta)&can
        r[r>0]-=1; r[fired]=1
        act=fired.astype(np.float32)*pol; cur[fired]=0.0
    return act,cur,r

def get_sparse(mask):
    rows,cols=np.where(mask)
    return rows.astype(np.intp),cols.astype(np.intp)

def eval_cosine(mask, theta, decay, pol, seqs, bp, ip, op, bigram):
    sp=get_sparse(mask)
    pn=bp/(np.linalg.norm(bp,axis=1,keepdims=True)+1e-8)
    tot=0.0
    for seq in seqs:
        st=np.zeros(H,dtype=np.float32);ch=np.zeros(H,dtype=np.float32)
        rf=np.zeros(H,dtype=np.int8);ss=0.0;n=0
        for i in range(len(seq)-1):
            inj=bp[seq[i]]@ip
            st,ch,rf=rollout(inj,mask,theta,decay,st,ch,sp,pol,rf)
            out=ch@op;on=out/(np.linalg.norm(out)+1e-8)
            sims=on@pn.T;e=np.exp(sims-sims.max());pred=e/e.sum()
            tgt=bigram[seq[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            ss+=cos;n+=1
        tot+=ss/n if n else 0.0
    return tot/len(seqs)

def eval_full(mask, theta, decay, pol, seqs, bp, ip, op, bigram):
    """Return cosine, accuracy, top1 prob, KL div."""
    sp=get_sparse(mask)
    pn=bp/(np.linalg.norm(bp,axis=1,keepdims=True)+1e-8)
    cosines=[];ok=0;tot=0;top1s=[];kls=[]
    for seq in seqs:
        st=np.zeros(H,dtype=np.float32);ch=np.zeros(H,dtype=np.float32)
        rf=np.zeros(H,dtype=np.int8)
        for i in range(len(seq)-1):
            inj=bp[seq[i]]@ip
            st,ch,rf=rollout(inj,mask,theta,decay,st,ch,sp,pol,rf)
            out=ch@op;on=out/(np.linalg.norm(out)+1e-8)
            sims=on@pn.T;e=np.exp(sims-sims.max());pred=e/e.sum()
            tgt=bigram[seq[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            cosines.append(cos)
            tgt_s=np.clip(tgt,1e-10,1);pred_s=np.clip(pred,1e-10,1)
            kls.append(np.sum(tgt_s*np.log(tgt_s/pred_s)))
            top1s.append(float(np.max(pred)))
            if np.argmax(pred)==seq[i+1]: ok+=1
            tot+=1
    return np.mean(cosines), ok/tot if tot else 0, np.mean(top1s), np.mean(kls)

def inject_random(mask, n, rng):
    added=0; att=0
    while added<n and att<n*20:
        r=rng.randint(0,H);c=rng.randint(0,H)
        if r!=c and not mask[r,c]: mask[r,c]=True; added+=1
        att+=1
    return added

def inject_loops(mask, loop_len, budget, rng):
    added=0; att=0
    while added<budget and att<budget*10:
        nodes=rng.choice(H,size=loop_len,replace=False).tolist()
        edges=[]
        ok=True
        for i in range(loop_len):
            r,c=nodes[i],nodes[(i+1)%loop_len]
            if mask[r,c]: ok=False; break
            edges.append((r,c))
        if ok and added+len(edges)<=budget:
            for r,c in edges: mask[r,c]=True; added+=1
        att+=1
    return added

def inject_mixed(mask, budget, rng):
    added=0; sizes=[3,4,5,6,7]; idx=0; att=0
    while added<budget and att<budget*10:
        sz=sizes[idx%len(sizes)]
        rem=budget-added
        if sz>rem:
            added+=inject_random(mask,rem,rng); break
        nodes=rng.choice(H,size=sz,replace=False).tolist()
        edges=[];ok=True
        for i in range(sz):
            r,c=nodes[i],nodes[(i+1)%sz]
            if mask[r,c]: ok=False; break
            edges.append((r,c))
        if ok and added+len(edges)<=budget:
            for r,c in edges: mask[r,c]=True; added+=1
            idx+=1
        att+=1
    return added

def run_arm(label, mask_init, theta, decay, pol, ip, op, bp, bigram, all_data, eval_seqs):
    rng=random.Random(SEED); np_rng=np.random.RandomState(SEED)
    mask=mask_init.copy()
    log=[]

    cos0,acc0,top1_0,kl0=eval_full(mask,theta,decay,pol,eval_seqs,bp,ip,op,bigram)
    edges0=int(np.sum(mask))
    log.append((0,cos0,acc0,top1_0,kl0,edges0))
    print(f'    [   0] cos={cos0:.6f} acc={acc0*100:.2f}% kl={kl0:.4f} top1={top1_0:.4f} e={edges0}')

    for step in range(1,TRAINING_STEPS+1):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        na=int(np.sum(mask))
        if pt in('flip','decay') and na==0: pt='add'
        nm=mask;nd=decay
        if pt=='add':
            r2=rng.randint(0,H-1);c2=rng.randint(0,H-1)
            if r2==c2 or mask[r2,c2]: continue
            nm=mask.copy();nm[r2,c2]=True
        elif pt=='flip':
            al=list(zip(*np.where(mask))) if na>0 else []
            if not al: continue
            r2,c2=al[rng.randint(0,len(al)-1)];nc2=rng.randint(0,H-1)
            if nc2==r2 or nc2==c2 or mask[r2,nc2]: continue
            nm=mask.copy();nm[r2,c2]=False;nm[r2,nc2]=True
        elif pt=='decay':
            idx=rng.randint(0,H-1);nd=decay.copy()
            nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
        tr=[all_data[o:o+SEQ_LEN] for o in [np_rng.randint(0,len(all_data)-SEQ_LEN) for _ in range(N_TRAIN)]]
        os_=eval_cosine(mask,theta,decay,pol,tr,bp,ip,op,bigram)
        ns_=eval_cosine(nm,theta,nd,pol,tr,bp,ip,op,bigram)
        if ns_-os_>THRESHOLD: mask=nm;decay=nd

        if step%REPORT_EVERY==0:
            cos,acc,top1,kl=eval_full(mask,theta,decay,pol,eval_seqs,bp,ip,op,bigram)
            edges=int(np.sum(mask))
            log.append((step,cos,acc,top1,kl,edges))
            print(f'    [{step:4d}] cos={cos:.6f} acc={acc*100:.2f}% kl={kl:.4f} top1={top1:.4f} e={edges}')
            sys.stdout.flush()
    return log

if __name__=='__main__':
    print('Loading...')
    all_data=load_fineweb_bytes()
    bigram=np.load(ROOT/'recipes'/'data'/'bigram_table.npy')
    bp=make_bp()
    net=SelfWiringGraph.load(str(CKPT))
    ip=net.input_projection;op=net.output_projection
    pol=np.where(net.polarity,1.0,-1.0).astype(np.float32)
    theta=net.theta.astype(np.float32);decay=net.decay
    base_mask=net.mask.copy()

    eval_rng=np.random.RandomState(9999)
    eval_seqs=[all_data[o:o+SEQ_LEN] for o in [eval_rng.randint(0,len(all_data)-SEQ_LEN) for _ in range(N_EVAL)]]

    ARMS=[
        ('baseline',  'none',   {}),
        ('random-30', 'random', {}),
        ('loop-3',    'loop',   {'length':3}),
        ('loop-5',    'loop',   {'length':5}),
        ('mixed-3-7', 'mixed',  {}),
    ]

    all_logs={}
    for label,inject_type,params in ARMS:
        print(f'\n  === {label} ===')
        mask=base_mask.copy()
        inject_rng=np.random.RandomState(SEED+hash(label)%10000)
        if inject_type=='random': inject_random(mask,INJECTION_BUDGET,inject_rng)
        elif inject_type=='loop': inject_loops(mask,params['length'],INJECTION_BUDGET,inject_rng)
        elif inject_type=='mixed': inject_mixed(mask,INJECTION_BUDGET,inject_rng)
        all_logs[label]=run_arm(label,mask,theta,decay,pol,ip,op,bp,bigram,all_data,eval_seqs)

    # Summary
    sep='='*70
    print(f'\n{sep}')
    print('  COSINE PROGRESSION SUMMARY')
    print(sep)
    print(f'  {"step":>5}', end='')
    for label,_,_ in ARMS: print(f'  {label:>12}', end='')
    print()

    steps_to_show=[0,100,300,500,1000,1500]
    for s in steps_to_show:
        print(f'  {s:>5}', end='')
        for label,_,_ in ARMS:
            log=all_logs[label]
            match=[e for e in log if e[0]==s]
            if match:
                print(f'  {match[0][1]:>12.6f}', end='')
            else:
                print(f'  {"---":>12}', end='')
        print()

    # Final comparison
    print(f'\n  Final (step {TRAINING_STEPS}):')
    for label,_,_ in ARMS:
        log=all_logs[label]
        final=log[-1]
        cos_start=log[0][1]
        cos_end=final[1]
        improvement=(cos_end-cos_start)/cos_start*100
        print(f'    {label:12s}: cos={cos_end:.6f} (+{improvement:.1f}%) acc={final[2]*100:.2f}% kl={final[4]:.4f} edges={final[5]}')
    print(sep)
