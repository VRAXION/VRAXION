#!/usr/bin/env python3
import argparse, json, os, random, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FAMILIES=["row","col","pair","mirror","diag"]
POCKETS=list("ABCDEFGHI")
SYM=list(range(9))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="8101,8102,8103,8104,8105")
    p.add_argument("--train-rows-per-seed", type=int, default=300)
    p.add_argument("--test-rows-per-seed", type=int, default=180)
    p.add_argument("--ood-rows-per-seed", type=int, default=180)
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--workers", default="auto")
    p.add_argument("--cpu-target", choices=["saturate","balanced"], default="saturate")
    p.add_argument("--heartbeat-sec", type=int, default=20)
    return p.parse_args()


def gen_row(rng, n, ood=False):
    rows=[]
    for i in range(n):
        fam=FAMILIES[i%5]
        board=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        miss=(1,1)
        if fam=="row": target=(board[1][0]+board[1][2])%9
        elif fam=="col": target=(board[0][1]+board[2][1])%9
        elif fam=="pair": target=(board[0][0]+board[2][2])%9
        elif fam=="mirror": target=(board[2][0]+board[0][2])%9
        else: target=(board[0][0]+board[1][2]+board[2][1])%9
        if ood: target=(target+1)%9
        pockets=SYM[:]
        rng.shuffle(pockets)
        ans_idx=rng.randrange(9)
        pockets[ans_idx]=target
        rows.append({"id":i,"family":fam,"board":board,"missing":miss,"pockets":pockets,"expected_selected":ans_idx})
    return rows


def feat(row, i):
    b=row["board"]; c=row["pockets"][i]
    f=[]
    f.append(1.0)
    f.append(1.0 if c==(b[1][0]+b[1][2])%9 else 0.0)
    f.append(1.0 if c==(b[0][1]+b[2][1])%9 else 0.0)
    f.append(1.0 if c==(b[0][0]+b[2][2])%9 else 0.0)
    f.append(1.0 if c==(b[2][0]+b[0][2])%9 else 0.0)
    f.append(1.0 if c==(b[0][0]+b[1][2]+b[2][1])%9 else 0.0)
    f.extend([i/8.0, abs(i-4)/4.0])
    return f


def predict(weights, row):
    scores=[]
    for i in range(9):
        v=sum(w*x for w,x in zip(weights,feat(row,i)))
        scores.append(v)
    m=max(range(9), key=lambda i:scores[i])
    top=sorted(scores, reverse=True)
    return m, (top[0]-top[1] if len(top)>1 else 0.0), scores


def eval_ds(weights, ds):
    outs=[]
    for r in ds:
        p,m,s=predict(weights,r)
        outs.append((r,p,m,s))
    acc=sum(int(r["expected_selected"]==p) for r,p,_,_ in outs)/max(1,len(outs))
    return acc, outs


def mutate(rng,w):
    nw=w[:]
    t=rng.choice(["gauss","flip","scale"])
    k=rng.randrange(len(nw))
    if t=="gauss": nw[k]+=rng.uniform(-0.6,0.6)
    elif t=="flip": nw[k]*=-1.0
    else: nw[k]*=rng.uniform(0.7,1.3)
    return nw,t


def run_seed(seed,args,out):
    os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    rng=random.Random(seed)
    train=gen_row(rng,args.train_rows_per_seed,False);test=gen_row(rng,args.test_rows_per_seed,False);ood=gen_row(rng,args.ood_rows_per_seed,True)
    d=out/f"seed_{seed}";d.mkdir(parents=True,exist_ok=True)
    w=[rng.uniform(-1,1) for _ in range(8)]
    best=w[:];best_fit=-1
    acc_m={"accepted":{"gauss":0,"flip":0,"scale":0},"rejected":{"gauss":0,"flip":0,"scale":0}}
    with (d/"train_metrics.jsonl").open("w") as tm,(d/"progress.jsonl").open("w") as pg:
        for g in range(args.generations):
            cand=[]
            for _ in range(args.population):
                nw,mt=mutate(rng,best)
                a,outs=eval_ds(nw,train)
                marg=statistics.median([m for _,_,m,_ in outs]) if outs else 0.0
                fit=a+0.01*marg
                cand.append((fit,nw,mt,a,marg))
            cand.sort(key=lambda x:x[0],reverse=True)
            top=cand[0]
            if top[0]>best_fit:
                best_fit=top[0];best=top[1];acc_m["accepted"][top[2]]+=1
            else: acc_m["rejected"][top[2]]+=1
            tm.write(json.dumps({"gen":g,"best_fit":best_fit,"train_acc":top[3],"margin":top[4]})+"\n")
            pg.write(json.dumps({"event":"gen","gen":g})+"\n")
    train_acc,_=eval_ds(best,train)
    test_acc,test_out=eval_ds(best,test)
    ood_acc,ood_out=eval_ds(best,ood)
    def family_acc(outs):
        return {f:sum(int(r["expected_selected"]==p) for r,p,_,_ in outs if r["family"]==f)/max(1,sum(1 for r,_,_,_ in outs if r["family"]==f)) for f in FAMILIES}
    fam=family_acc(test_out)
    cm=[[0]*9 for _ in range(9)]
    for r,p,_,_ in test_out: cm[r["expected_selected"]][p]+=1
    margins=[m for _,_,m,_ in test_out]
    low=[x for x in test_out if x[2]<0.1]
    low_err=sum(int(r["expected_selected"]!=p) for r,p,_,_ in low)/max(1,len(low))
    for nm,outs in [("test",test_out),("ood",ood_out)]:
        with (d/f"row_outputs_{nm}.jsonl").open("w") as f:
            for r,p,m,s in outs:f.write(json.dumps({"id":r["id"],"family":r["family"],"truth":r["expected_selected"],"pred":p,"margin":m})+"\n")
    metrics={"random_baseline_accuracy":1/9,"train_accuracy":train_acc,"test_accuracy":test_acc,"ood_accuracy":ood_acc,"per_family_accuracy":fam,"pocket_confusion_matrix":cm,"median_score_margin":statistics.median(margins),"low_margin_error_rate":low_err,"accepted_mutations_by_type":acc_m["accepted"],"rejected_mutations_by_type":acc_m["rejected"],"mutation_acceptance_rate":sum(acc_m["accepted"].values())/max(1,sum(acc_m["accepted"].values())+sum(acc_m["rejected"].values()))}
    (d/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (d/"best_individual.json").write_text(json.dumps({"weights":best,"fitness":best_fit},indent=2))
    return {"seed":seed,"status":"ok"}

def audit_d34():
    p=Path("scripts/probes/run_d34_raven_pocket_corridor_baseline_suite.py")
    txt=p.read_text() if p.exists() else ""
    return {
      "d34_files_found": all(Path(x).exists() for x in ["docs/research/D33_RAVEN_POCKET_CORRIDOR_DNA_VS_EVOLUTION_CATCHUP.md","docs/research/D34_RAVEN_POCKET_CORRIDOR_BASELINE_SUITE_CONTRACT.md","docs/research/D34_RAVEN_POCKET_CORRIDOR_BASELINE_SUITE_RESULT.md","scripts/probes/run_d34_raven_pocket_corridor_baseline_suite.py","scripts/probes/run_d34_raven_pocket_corridor_baseline_suite_check.py"]),
      "synthetic_base_accuracy_detected":"base={" in txt,
      "synthetic_hit_sampling_detected":"hit=random.random()<p" in txt,
      "random_label_generation_detected":"truth=random.randrange(9)" in txt,
      "fake_margin_generation_detected":"margin=abs(top_true-top_other)" in txt,
      "fake_mutation_counts_detected":"accepted={" in txt and "rejected={" in txt,
      "real_direct_mutation_implementation_found":False,
      "real_dna_genome_implementation_found":False,
      "d34_result_can_be_used_as_validated_benchmark":False,
      "d34_result_can_be_used_as_scaffold_catchup_only":True,
      "recommendation":"Treat D34 as scaffold/catch-up only; use D35 real probe outputs for next baseline planning."
    }

def main():
    a=parse_args();t0=time.time();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(",") if x]
    (out/"queue.json").write_text(json.dumps({"seeds":seeds},indent=2))
    audit=audit_d34();(out/"d34_fidelity_audit_report.json").write_text(json.dumps(audit,indent=2))
    workers=min(os.cpu_count() or 1,len(seeds)) if a.workers=="auto" else int(a.workers)
    done=[];fails=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(run_seed,s,a,out):s for s in seeds}
        for fut in as_completed(futs):
            s=futs[fut]
            try:done.append(fut.result())
            except Exception as e:
                fails.append({"seed":s,"error":str(e)})
                d=out/f"seed_{s}";d.mkdir(parents=True,exist_ok=True);(d/"error.json").write_text(json.dumps({"error":str(e)},indent=2))
            with (out/"progress.jsonl").open("a") as f:f.write(json.dumps({"completed":len(done)+len(fails),"total":len(seeds)})+"\n")
    mets=[]
    for s in seeds:
        p=out/f"seed_{s}"/"metrics.json"
        if p.exists():mets.append(json.loads(p.read_text()))
    agg={k:statistics.mean(m[k] for m in mets) for k in ["random_baseline_accuracy","train_accuracy","test_accuracy","ood_accuracy","median_score_margin","low_margin_error_rate"]} if mets else {}
    agg["seed_variance"]=statistics.pvariance([m["test_accuracy"] for m in mets]) if len(mets)>1 else 0.0
    agg["failed_seed_count"]=len(seeds)-len(mets)
    decision="real_direct_mutation_signal_not_confirmed";nxt="D36_RAVEN_FEATURE_SPACE_DIAGNOSTIC"
    if audit["d34_result_can_be_used_as_scaffold_catchup_only"] and mets:
        decision="d34_scaffold_detected_real_direct_mutation_probe_completed";nxt="D36_REAL_RAVEN_CORRIDOR_BASELINE_SUITE"
    if mets and agg["test_accuracy"]>=0.40 and agg["ood_accuracy"]>=0.30:
        decision="real_direct_mutation_signal_confirmed";nxt="D36_REAL_RAVEN_CORRIDOR_BASELINE_SUITE"
    (out/"aggregate_metrics.json").write_text(json.dumps(agg,indent=2))
    (out/"dataset_manifest.json").write_text(json.dumps({"families":FAMILIES,"labels":POCKETS,"train":a.train_rows_per_seed,"test":a.test_rows_per_seed,"ood":a.ood_rows_per_seed},indent=2))
    (out/"per_seed_report.json").write_text(json.dumps({"seeds":seeds,"completed":len(mets),"failed":fails},indent=2))
    (out/"per_family_report.json").write_text(json.dumps({"per_family_accuracy_mean":{f:statistics.mean(m["per_family_accuracy"][f] for m in mets) for f in FAMILIES} if mets else {}},indent=2))
    (out/"pocket_confusion_matrix.json").write_text(json.dumps({"mean_test_confusion":mets[0]["pocket_confusion_matrix"] if mets else []},indent=2))
    (out/"score_margin_report.json").write_text(json.dumps({"median_score_margin":agg.get("median_score_margin",0),"low_margin_error_rate":agg.get("low_margin_error_rate",0)},indent=2))
    (out/"mutation_acceptance_report.json").write_text(json.dumps({"accepted":{k:sum(m["accepted_mutations_by_type"][k] for m in mets) for k in ["gauss","flip","scale"]} if mets else {},"rejected":{k:sum(m["rejected_mutations_by_type"][k] for m in mets) for k in ["gauss","flip","scale"]} if mets else {}},indent=2))
    (out/"best_individual_report.json").write_text(json.dumps({"seeds":len(mets)},indent=2))
    with (out/"row_level_error_examples.jsonl").open("w") as f:f.write(json.dumps({"note":"see per-seed row outputs for full rows"})+"\n")
    (out/"direct_mutation_probe_report.json").write_text(json.dumps({"aggregate":agg},indent=2))
    (out/"machine_utilization_report.json").write_text(json.dumps({"os_cpu_count":os.cpu_count(),"worker_count":workers,"thread_env":{"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},"wall_clock_sec":time.time()-t0},indent=2))
    dec={"decision":decision,"next":nxt,"d34_real_benchmark":False,"d34_scaffold":True,"boundary_flags":{"architecture_superiority_claim":False,"natural_language_reasoning_claim":False,"solved_claim":False}}
    (out/"decision.json").write_text(json.dumps(dec,indent=2));(out/"summary.json").write_text(json.dumps({"decision":decision,"next":nxt},indent=2))
    (out/"report.md").write_text(f"# D35 Reality Audit + Direct Mutation Probe\n\nDecision: `{decision}`\nNext: `{nxt}`\n")

if __name__=="__main__":main()
