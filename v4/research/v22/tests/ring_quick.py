"""
RING BUFFER — minimal test. Graph only vs Graph+Ring on 3 tasks.
"""
import numpy as np, time, math, random

def softmax(x):
    e=np.exp(x-x.max());return e/e.sum()

class GraphRing:
    def __init__(self,N,n_in,n_out,ring_slots=0,slot_dim=8):
        self.N=N;self.n_in=n_in;self.n_out=n_out;self.ring_slots=ring_slots;self.slot_dim=slot_dim
        s=math.sqrt(2.0/N)
        self.W=np.random.randn(N,N)*s
        self.mask=(np.random.rand(N,N)<0.06).astype(float);np.fill_diagonal(self.mask,0)
        self.addr=np.random.randn(N,4);self.tw=np.random.randn(N,4)*0.1
        self.state=np.zeros(N);self.decay=0.5;self.last_acc=0.0
        if ring_slots>0:
            self.ring=np.zeros((ring_slots,slot_dim));self.rp=0
            self.Wq=np.random.randn(N,slot_dim)*0.1
            self.Ww=np.random.randn(N,slot_dim)*0.1
            self.Wi=np.random.randn(slot_dim,N)*0.1
    def reset(self):
        self.state=np.zeros(self.N)
        if self.ring_slots>0:self.ring=np.zeros((self.ring_slots,self.slot_dim));self.rp=0
    def forward(self,world,diff,ticks=6):
        inp=np.concatenate([world,diff]);act=self.state.copy();Weff=self.W*self.mask
        if self.ring_slots>0:
            q=act@self.Wq;sc=self.ring@q;se=np.exp(sc-sc.max());at=se/(se.sum()+1e-8)
            rc=(at@self.ring)@self.Wi
        else:rc=0
        for t in range(ticks):
            act=act*self.decay;act[:self.n_in]=inp
            if t==0:act=act+rc*0.3 if self.ring_slots>0 else act
            raw=act@Weff+act*0.1;act=np.where(raw>0,raw,0.01*raw);act[:self.n_in]=inp
        self.state=act.copy()
        if self.ring_slots>0:
            self.ring[self.rp]=act@self.Ww;self.rp=(self.rp+1)%self.ring_slots
        # sw inverse
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
    def save(self):
        b=[self.W.copy(),self.mask.copy(),self.state.copy(),self.addr.copy(),self.tw.copy()]
        if self.ring_slots>0:b+=[self.ring.copy(),self.Wq.copy(),self.Ww.copy(),self.Wi.copy(),self.rp]
        return tuple(b)
    def restore(self,s):
        self.W,self.mask,self.state,self.addr,self.tw=s[0].copy(),s[1].copy(),s[2].copy(),s[3].copy(),s[4].copy()
        if self.ring_slots>0:self.ring,self.Wq,self.Ww,self.Wi,self.rp=s[5].copy(),s[6].copy(),s[7].copy(),s[8].copy(),s[9]
    def mut_struct(self,rate=0.05):
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
    def mut_weights(self,scale=0.05):
        self.W+=np.random.randn(*self.W.shape)*scale*self.mask
        self.tw+=np.random.randn(*self.tw.shape)*scale*0.5
        self.addr+=np.random.randn(*self.addr.shape)*scale*0.2
        if self.ring_slots>0:
            self.Wq+=np.random.randn(*self.Wq.shape)*scale*0.3
            self.Ww+=np.random.randn(*self.Ww.shape)*scale*0.3
            self.Wi+=np.random.randn(*self.Wi.shape)*scale*0.3

def run(label,task,ring,v=8,max_att=3000,seed=42):
    np.random.seed(seed);random.seed(seed)
    n_in=v*2;n_out=v;N=n_in+24+n_out
    net=GraphRing(N,n_in,n_out,ring,8)

    if task=="lookup":
        perm=np.random.permutation(v);inputs=list(range(v))
        def ev():
            net.reset();pd=np.zeros(v);c=0
            for p in range(2):
                for i in range(v):
                    w=np.zeros(v);w[inputs[i]]=1.0;lo=net.forward(w,pd)
                    pr=softmax(lo[:v])
                    if p==1 and np.argmax(pr)==perm[i]:c+=1
                    tv=np.zeros(v);tv[int(perm[i])]=1.0;pd=tv-pr
            return c/v
    else:
        seqs=[]
        for _ in range(3):
            si=[random.randint(0,v-1) for _ in range(12)]
            if task=="sequence":st=[0]+[(si[i-1]+si[i])%v for i in range(1,12)]
            else:st=[0,0]+[si[i-2] for i in range(2,12)]
            seqs.append((si,st))
        def ev():
            tc=0;ti=0
            for si,st in seqs:
                net.reset();pd=np.zeros(v)
                for i in range(len(si)):
                    w=np.zeros(v);w[si[i]]=1.0;lo=net.forward(w,pd)
                    pr=softmax(lo[:v])
                    if i>1:
                        if np.argmax(pr)==st[i]:tc+=1
                        ti+=1
                    tv=np.zeros(v);tv[st[i]]=1.0;pd=tv-pr
            return tc/max(1,ti)

    score=ev();best=score;phase="S";kept=0;stale=0;sw=False;t0=time.time()
    for att in range(max_att):
        s=net.save()
        if phase=="S":net.mut_struct(0.05)
        else:
            if random.random()<0.3:net.mut_struct(0.05)
            else:net.mut_weights(0.05)
        ns=ev();net.last_acc=ns
        if ns>score:score=ns;kept+=1;stale=0;best=max(best,score)
        else:net.restore(s);stale+=1
        if phase=="S" and stale>1500 and not sw:phase="B";sw=True;stale=0
        if best>=0.99:break
        if stale>=5000:break
    return {"label":label,"acc":best,"att":att+1,"time":time.time()-t0,"conns":net.conns(),"kept":kept}

print("="*60)
print("RING BUFFER TEST — 8-class, small & fast")
print("="*60)

for task,desc in [("lookup","A→B Lookup"),("sequence","Prev+Curr mod 8"),("repeat","Repeat-2-ago")]:
    print(f"\n--- {desc} ---")
    print(f"  {'Config':<20} {'Acc':>5} {'Conns':>6} {'Kept':>5} {'Time':>5}")
    for label,ring in [("Graph only",0),("Graph+Ring4",4),("Graph+Ring8",8)]:
        r=run(label,task,ring)
        print(f"  {label:<20} {r['acc']*100:4.0f}% {r['conns']:>6} {r['kept']:>5} {r['time']:>4.0f}s")

print(f"\nRandom: Lookup={100/8:.0f}%, Sequence={100/8:.0f}%, Repeat={100/8:.0f}%")
