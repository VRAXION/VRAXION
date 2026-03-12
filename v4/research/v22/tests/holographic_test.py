"""
DENSE DISTRIBUTED CODING TEST
"""
import numpy as np, time, math, random

def softmax(x):
    e=np.exp(x-x.max());return e/e.sum()

class Net:
    def __init__(self,N,n_in,n_out,density=0.06,mode="baseline",part=0.5):
        self.N=N;self.n_in=n_in;self.n_out=n_out;self.mode=mode;self.part=part;self.last_acc=0.0
        s=math.sqrt(2.0/N);self.W=np.random.randn(N,N)*s
        self.mask=(np.random.rand(N,N)<density).astype(float);np.fill_diagonal(self.mask,0)
        self.addr=np.random.randn(N,4);self.tw=np.random.randn(N,4)*0.1
        self.state=np.zeros(N);self.decay=0.5
    def _dist(self,act):
        s=slice(self.n_in,self.N-self.n_out);v=act[s].copy();n=len(v)
        if self.mode=="baseline":return act
        elif self.mode=="normalize":
            nm=np.sqrt((v**2).sum()+1e-8)
            if nm>0.01:act[s]=v/nm*math.sqrt(n)*0.1
        elif self.mode=="tanh_spread":act[s]=np.tanh(v*2)*0.5
        elif self.mode=="abs_tanh":act[s]=np.tanh(np.abs(v)*2)*0.5
        elif self.mode=="group_multi":
            gs=max(3,n//5)
            for g in range(0,n,gs):
                e=min(g+gs,n);grp=v[g:e]
                if len(grp)==0:continue
                med=np.median(np.abs(grp))
                d=np.where(np.abs(grp)>=med,1.0,0.3)
                v[g:e]=grp*d
            act[s]=v
        elif self.mode=="soft_all":
            # Softmax-style: everyone participates proportionally
            av=np.abs(v)+1e-8;w=av/av.sum();act[s]=np.sign(v)*w*n*0.3
        return act
    def reset(self):self.state=np.zeros(self.N)
    def forward(self,world,diff,ticks=6):
        inp=np.concatenate([world,diff]);act=self.state.copy();Weff=self.W*self.mask
        for t in range(ticks):
            act=act*self.decay;act[:self.n_in]=inp
            raw=act@Weff+act*0.1;act=np.where(raw>0,raw,0.01*raw)
            act=self._dist(act);act[:self.n_in]=inp
        self.state=act.copy()
        if self.last_acc<0.3:tk,mn=2,1
        elif self.last_acc<0.7:tk,mn=3,2
        else:tk,mn=5,3
        a2=np.abs(act[self.n_in:])
        if a2.sum()>0.01:
            nc=min(tk,len(a2));top=np.argpartition(a2,-nc)[-nc:]+self.n_in;new=0
            for ni in top:
                ni=int(ni)
                if np.abs(act[ni])<0.1:continue
                tgt=self.addr[ni]+np.abs(act[ni])*self.tw[ni]
                d=((self.addr-tgt)**2).sum(axis=1);d[ni]=1e9;near=int(np.argmin(d))
                if self.mask[ni,near]==0:self.mask[ni,near]=1;self.W[ni,near]=random.gauss(0,math.sqrt(2.0/self.N));new+=1
                if new>=mn:break
        return act[-self.n_out:]
    def conns(self):return int(self.mask.sum())
    def save(self):return(self.W.copy(),self.mask.copy(),self.state.copy(),self.addr.copy(),self.tw.copy())
    def restore(self,s):self.W,self.mask,self.state,self.addr,self.tw=s[0].copy(),s[1].copy(),s[2].copy(),s[3].copy(),s[4].copy()
    def mutS(self,rate=0.05):
        a=random.choice(["add","remove","rewire"])
        if a=="add":
            d=np.argwhere(self.mask==0);d=d[d[:,0]!=d[:,1]]
            if len(d)>0:
                n=max(1,int(len(d)*rate));idx=d[np.random.choice(len(d),min(n,len(d)),replace=False)]
                for j in range(len(idx)):self.mask[int(idx[j][0]),int(idx[j][1])]=1;self.W[int(idx[j][0]),int(idx[j][1])]=random.gauss(0,math.sqrt(2.0/self.N))
        elif a=="remove":
            al=np.argwhere(self.mask==1)
            if len(al)>3:
                n=max(1,int(len(al)*rate));idx=al[np.random.choice(len(al),min(n,len(al)),replace=False)]
                for j in range(len(idx)):self.mask[int(idx[j][0]),int(idx[j][1])]=0
        else:
            al=np.argwhere(self.mask==1)
            if len(al)>0:
                n=max(1,int(len(al)*rate));idx=al[np.random.choice(len(al),min(n,len(al)),replace=False)]
                for j in range(len(idx)):
                    r,c=int(idx[j][0]),int(idx[j][1]);self.mask[r,c]=0;nc=random.randint(0,self.N-1)
                    while nc==r:nc=random.randint(0,self.N-1)
                    self.mask[r,nc]=1;self.W[r,nc]=self.W[r,c]
    def mutW(self,scale=0.05):
        self.W+=np.random.randn(*self.W.shape)*scale*self.mask
        self.tw+=np.random.randn(*self.tw.shape)*scale*0.5
        self.addr+=np.random.randn(*self.addr.shape)*scale*0.2

def run(label,mode,part=0.5,v=16,N=80,max_att=4000,seed=42):
    np.random.seed(seed);random.seed(seed)
    perm=np.random.permutation(v);inputs=list(range(v))
    n_in=v*2;n_out=v;ticks=6
    net=Net(N,n_in,n_out,0.06,mode,part)
    def ev():
        net.reset();pd=np.zeros(v);c=0
        for p in range(2):
            for i in range(v):
                w=np.zeros(v);w[inputs[i]]=1.0;lo=net.forward(w,pd)
                pr=softmax(lo[:v])
                if p==1 and np.argmax(pr)==perm[i]:c+=1
                tv=np.zeros(v);tv[int(perm[i])]=1.0;pd=tv-pr
        return c/v
    def patterns():
        pats=[]
        for i in range(v):
            net.reset();w=np.zeros(v);w[i]=1.0;pd=np.zeros(v);net.forward(w,pd)
            pats.append(net.state[n_in:N-n_out].copy())
        pats=np.array(pats)
        n_act=np.mean([(np.abs(p)>0.01).sum() for p in pats])
        norms=np.sqrt((pats**2).sum(axis=1,keepdims=True)+1e-8)
        normed=pats/norms;sim=normed@normed.T
        mask_d=1-np.eye(v);avg_sim=(np.abs(sim)*mask_d).sum()/mask_d.sum()
        return n_act,avg_sim

    score=ev();best=score;phase="S";kept=0;stale=0;sw=False;t0=time.time();curve=[]
    for att in range(max_att):
        s=net.save()
        if phase=="S":net.mutS()
        else:
            if random.random()<0.3:net.mutS()
            else:net.mutW()
        ns=ev();net.last_acc=ns
        if ns>score:score=ns;kept+=1;stale=0;best=max(best,score)
        else:net.restore(s);stale+=1
        if phase=="S" and stale>2000 and not sw:phase="B";sw=True;stale=0
        if (att+1)%500==0:curve.append((att+1,score))
        if best>=0.99:break
        if stale>=6000:break
    n_act,avg_sim=patterns()
    return{"label":label,"acc":best,"att":att+1,"time":time.time()-t0,
           "conns":net.conns(),"curve":curve,"n_act":n_act,"sim":avg_sim}

print("="*65)
print("DISTRIBUTED vs SPARSE CODING")
print("="*65)
print("16-class, 80 neurons\n")
configs=[
    ("Baseline",       "baseline"),
    ("L2 normalize",   "normalize"),
    ("tanh spread",    "tanh_spread"),
    ("abs+tanh",       "abs_tanh"),
    ("Group multi-win","group_multi"),
    ("Soft-all prop",  "soft_all"),
]
results=[]
for label,mode in configs:
    print(f"  {label:20s}...",end=" ",flush=True)
    r=run(label,mode);results.append(r)
    f100="never"
    for s,a in r['curve']:
        if a>=0.99:f100=str(s);break
    print(f"{r['acc']*100:.0f}% | 100%@{f100:>5} | "
          f"Active:{r['n_act']:.1f} | Overlap:{r['sim']:.3f} | "
          f"Conns:{r['conns']} | {r['time']:.0f}s")

print(f"\n{'='*65}")
print(f"RANKED")
print(f"{'='*65}")
print(f"{'Config':<20} {'Acc':>5} {'100%@':>6} {'Active':>7} {'Overlap':>8} {'Conns':>6}")
print(f"{'-'*20} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")
def sk(r):
    for s,a in r['curve']:
        if a>=0.99:return(0,s)
    return(1,-r['acc'])
for r in sorted(results,key=sk):
    f100="never"
    for s,a in r['curve']:
        if a>=0.99:f100=str(s);break
    print(f"{r['label']:<20} {r['acc']*100:4.0f}% {f100:>6} "
          f"{r['n_act']:>6.1f} {r['sim']:>8.3f} {r['conns']:>6}")

print(f"\nKEY: Low overlap = orthogonal (good)")
print(f"     High active = distributed (holographic)")
print(f"     IDEAL = high active + low overlap")
