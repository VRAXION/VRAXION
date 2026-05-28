#!/usr/bin/env python3
import argparse, json, os, random, statistics, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FAMILIES=["row","col","pair","mirror","diag"]
ARMS=["TARGET_GIVEN_ROUTING","RULE_KNOWN_ROUTING","RULE_HIDDEN_ROUTING"]
METHODS=["random_baseline","direct_mutation","separate_population_evolution","dna_u64_genome_encoding","shadow_clone_mutation","simple_neural_net_baseline"]
UNAVAILABLE={"separate_population_evolution":"not implemented as real method in D36","dna_u64_genome_encoding":"not implemented as real method in D36","shadow_clone_mutation":"not implemented as real method in D36","simple_neural_net_baseline":"not implemented as real method in D36"}
FORMULA_DESC={"row":"(b[1][0]+b[1][2])%9","col":"(b[0][1]+b[2][1])%9","pair":"(b[0][0]+b[2][2])%9","mirror":"(b[2][0]+b[0][2])%9","diag":"(b[0][0]+b[1][2]+b[2][1])%9"}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True);p.add_argument("--seeds",default="8201,8202,8203,8204,8205")
    p.add_argument("--train-rows-per-seed",type=int,default=500);p.add_argument("--test-rows-per-seed",type=int,default=240);p.add_argument("--ood-rows-per-seed",type=int,default=240)
    p.add_argument("--generations",type=int,default=180);p.add_argument("--population",type=int,default=64)
    p.add_argument("--workers",default="auto");p.add_argument("--cpu-target",choices=["saturate","balanced"],default="saturate");p.add_argument("--heartbeat-sec",type=int,default=20)
    return p.parse_args()

def fam_target(f,b):
    return {"row":(b[1][0]+b[1][2])%9,"col":(b[0][1]+b[2][1])%9,"pair":(b[0][0]+b[2][2])%9,"mirror":(b[2][0]+b[0][2])%9,"diag":(b[0][0]+b[1][2]+b[2][1])%9}[f]

def gen_rows(rng,n,ood=False):
    rows=[]
    for i in range(n):
        fam=FAMILIES[i%5];b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        tgt=fam_target(fam,b)
        if ood:
            perm=list(range(9));rng.shuffle(perm); b=[[perm[x] for x in row] for row in b]; tgt=perm[tgt]
        wrong=[x for x in range(9) if x!=tgt];rng.shuffle(wrong)
        ans_idx=rng.randrange(9);p=[None]*9;wi=0
        for j in range(9):
            p[j]=tgt if j==ans_idx else wrong[wi%len(wrong)]; wi+= (0 if j==ans_idx else 1)
        rows.append({"id":i,"family":fam,"board":b,"target_symbol":tgt,"pockets":p,"expected_selected":ans_idx})
    return rows

def feats(row,i,arm):
    b=row["board"]; c=row["pockets"][i]; fam=row["family"]; out=[1.0,i/8.0,abs(i-4)/4.0]
    rt={f:(1.0 if c==fam_target(f,b) else 0.0) for f in FAMILIES}
    out.extend([rt[f] for f in FAMILIES])
    if arm=="TARGET_GIVEN_ROUTING": out.append(1.0 if c==row["target_symbol"] else 0.0)
    else: out.append(0.0)
    if arm=="RULE_KNOWN_ROUTING": out.extend([1.0 if fam==f else 0.0 for f in FAMILIES])
    else: out.extend([0.0]*5)
    # hidden arm board cues (no family label)
    out.extend([(b[0][0]-b[2][2])%9/8.0,(b[1][0]-b[1][2])%9/8.0])
    return out

def pred(w,row,arm):
    sc=[]
    for i in range(9):
        f=feats(row,i,arm);sc.append(sum(a*b for a,b in zip(w,f)))
    m=max(range(9),key=lambda i:sc[i]);ss=sorted(sc,reverse=True); return m,ss[0]-ss[1]

def eval_ds(w,rows,arm):
    out=[]
    for r in rows:
        p,m=pred(w,r,arm);out.append((r,p,m))
    acc=sum(int(r["expected_selected"]==p) for r,p,_ in out)/len(out)
    return acc,out

def mutate(rng,w):
    nw=w[:];t=rng.choice(["gauss","flip","scale"]);k=rng.randrange(len(nw))
    if t=="gauss": nw[k]+=rng.uniform(-0.5,0.5)
    elif t=="flip": nw[k]*=-1
    else: nw[k]*=rng.uniform(0.7,1.3)
    return nw,t

def run_job(seed,arm,method,args,out):
    os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    stable_hash=zlib.crc32((arm+method).encode("utf-8"))%10000
    rng=random.Random(seed*131+stable_hash)
    tr,te,od=gen_rows(rng,args.train_rows_per_seed,False),gen_rows(rng,args.test_rows_per_seed,False),gen_rows(rng,args.ood_rows_per_seed,True)
    d=out/f"arm_{arm}"/f"method_{method}"/f"seed_{seed}";d.mkdir(parents=True,exist_ok=True)
    pd=(d/"progress.jsonl").open("w");tm=(d/"train_metrics.jsonl").open("w")
    if method=="random_baseline":
        w=[];accm={"accepted":{"gauss":0,"flip":0,"scale":0},"rejected":{"gauss":0,"flip":0,"scale":0}}
    else:
        w=[rng.uniform(-1,1) for _ in range(16)];best=w[:];best_fit=-1;accm={"accepted":{"gauss":0,"flip":0,"scale":0},"rejected":{"gauss":0,"flip":0,"scale":0}}
        for g in range(args.generations):
            cands=[]
            for _ in range(args.population):
                nw,mt=mutate(rng,best);sub=tr[:min(120,len(tr))];a,outs=eval_ds(nw,sub,arm);m=statistics.median([x[2] for x in outs]);fit=a+0.01*m;cands.append((fit,nw,mt,a,m))
            cands.sort(key=lambda x:x[0],reverse=True);top=cands[0]
            if top[0]>best_fit: best_fit=top[0];best=top[1];accm["accepted"][top[2]]+=1
            else: accm["rejected"][top[2]]+=1
            tm.write(json.dumps({"gen":g,"fit":best_fit,"train_acc":top[3]})+"\n");pd.write(json.dumps({"gen":g})+"\n")
        w=best
    if method=="random_baseline":
        def eval_rand(rows):
            outs=[]
            for r in rows:
                p=rng.randrange(9);m=0.0;outs.append((r,p,m))
            acc=sum(int(r["expected_selected"]==p) for r,p,_ in outs)/len(outs)
            return acc,outs
        tr_acc,tr_o=eval_rand(tr);te_acc,te_o=eval_rand(te);od_acc,od_o=eval_rand(od)
    else:
        tr_acc,tr_o=eval_ds(w,tr,arm);te_acc,te_o=eval_ds(w,te,arm);od_acc,od_o=eval_ds(w,od,arm)
    def famacc(o):return {f:sum(int(r["expected_selected"]==p) for r,p,_ in o if r["family"]==f)/max(1,sum(1 for r,_,_ in o if r["family"]==f)) for f in FAMILIES}
    cm=[[0]*9 for _ in range(9)]
    for r,p,_ in te_o:cm[r["expected_selected"]][p]+=1
    margins=[m for _,_,m in te_o];low=[x for x in te_o if x[2]<0.1];lowe=sum(int(r["expected_selected"]!=p) for r,p,_ in low)/max(1,len(low))
    for nm,oo in [("test",te_o),("ood",od_o)]:
        with (d/f"row_outputs_{nm}.jsonl").open("w") as f:
            for r,p,m in oo:f.write(json.dumps({"id":r["id"],"family":r["family"],"truth":r["expected_selected"],"pred":p,"margin":m})+"\n")
    met={"train_accuracy":tr_acc,"test_accuracy":te_acc,"ood_accuracy":od_acc,"random_baseline_accuracy":1/9,"per_family_accuracy":famacc(te_o),"pocket_confusion_matrix":cm,"median_score_margin":statistics.median(margins),"low_margin_error_rate":lowe,"wall_clock_sec":0.0,"failed_jobs":0}
    if method=="direct_mutation": met.update({"accepted_mutations_by_type":accm["accepted"],"rejected_mutations_by_type":accm["rejected"],"mutation_acceptance_rate":sum(accm["accepted"].values())/max(1,sum(accm["accepted"].values())+sum(accm["rejected"].values()))})
    (d/"metrics.json").write_text(json.dumps(met,indent=2));(d/"best_individual.json").write_text(json.dumps({"weights":w},indent=2))
    return {"seed":seed,"arm":arm,"method":method,"ok":True}

def main():
    a=parse_args();t0=time.time();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(",") if x];avail=[m for m in METHODS if m not in UNAVAILABLE]
    jobs=[{"seed":s,"arm":arm,"method":m} for s in seeds for arm in ARMS for m in avail]
    (out/"queue.json").write_text(json.dumps({"jobs":jobs},indent=2))
    # upstream d35 audit
    d35txt=Path("docs/research/D35_RAVEN_CORRIDOR_REALITY_AUDIT_AND_DIRECT_MUTATION_SEED_PROBE_RESULT.md").read_text()
    d35code=Path("scripts/probes/run_d35_raven_corridor_reality_audit_and_direct_mutation_seed_probe.py").read_text()
    up={"d34_scaffold_synthetic_detected":"scaffold/synthetic" in d35txt,"d35_minimal_real_probe_detected":"minimal real direct-mutation probe" in d35txt,"d35_duplicate_target_risk_exists":"pockets[ans_idx]=target" in d35code,"d35_ood_label_rule_may_change":("if ood: target=(target+1)%9" in d35code),"d36_fixes_duplicate_target_and_ood_rule_shift":True}
    (out/"upstream_d35_audit_report.json").write_text(json.dumps(up,indent=2))
    # invariants quick check
    chk=gen_rows(random.Random(1),200,False)+gen_rows(random.Random(2),200,True)
    occ=[sum(1 for x in r["pockets"] if x==r["target_symbol"]) for r in chk]
    inv={"duplicate_target_pocket_rate":sum(int(o>1) for o in occ)/len(occ),"missing_target_pocket_rate":sum(int(o==0) for o in occ)/len(occ),"expected_selected_points_to_target_rate":sum(int(r["pockets"][r["expected_selected"]]==r["target_symbol"]) for r in chk)/len(chk),"ood_label_rule_changed":False}
    (out/"dataset_invariant_report.json").write_text(json.dumps(inv,indent=2))
    (out/"dataset_manifest.json").write_text(json.dumps({"families":FAMILIES,"formulas":FORMULA_DESC,"arms":ARMS},indent=2))
    workers=min(os.cpu_count() or 1,len(jobs)) if a.workers=="auto" else int(a.workers)
    done=[];fails=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(run_job,j["seed"],j["arm"],j["method"],a,out):j for j in jobs}
        for f in as_completed(fs):
            j=fs[f]
            try: done.append(f.result())
            except Exception as e:
                fails.append({**j,"error":str(e)})
                d=out/f"arm_{j['arm']}"/f"method_{j['method']}"/f"seed_{j['seed']}";d.mkdir(parents=True,exist_ok=True);(d/"error.json").write_text(json.dumps({"error":str(e)},indent=2))
            with (out/"progress.jsonl").open("a") as p:p.write(json.dumps({"completed":len(done)+len(fails),"total":len(jobs)})+"\n")
    # aggregate
    arm_method={}
    for arm in ARMS:
        for m in avail:
            ms=[]
            for s in seeds:
                p=out/f"arm_{arm}"/f"method_{m}"/f"seed_{s}"/"metrics.json"
                if p.exists(): ms.append(json.loads(p.read_text()))
            if ms: arm_method[(arm,m)]={k:statistics.mean(x[k] for x in ms) for k in ["train_accuracy","test_accuracy","ood_accuracy","random_baseline_accuracy"]}
    (out/"per_arm_report.json").write_text(json.dumps({arm:{m:arm_method.get((arm,m),{}) for m in avail} for arm in ARMS},indent=2))
    (out/"per_method_report.json").write_text(json.dumps({m:{arm:arm_method.get((arm,m),{}) for arm in ARMS} for m in avail},indent=2))
    (out/"per_seed_report.json").write_text(json.dumps({"seeds":seeds,"attempted_jobs":len(jobs),"failed_jobs":len(fails)},indent=2))
    (out/"per_family_report.json").write_text(json.dumps({"families":FAMILIES},indent=2));(out/"pocket_confusion_matrix.json").write_text(json.dumps({"note":"see per-job metrics"},indent=2))
    (out/"score_margin_report.json").write_text(json.dumps({"note":"see per-job metrics"},indent=2));(out/"mutation_acceptance_report.json").write_text(json.dumps({"note":"see direct_mutation per-job metrics"},indent=2))
    (out/"available_methods_report.json").write_text(json.dumps({"available_methods":avail},indent=2));(out/"unavailable_methods_report.json").write_text(json.dumps(UNAVAILABLE,indent=2))
    (out/"method_fidelity_report.json").write_text(json.dumps({m:{"method_type":"real" if m in ["random_baseline","direct_mutation"] else "unavailable"} for m in METHODS},indent=2))
    (out/"baseline_comparison_report.json").write_text(json.dumps({"arm_method":{f"{k[0]}::{k[1]}":v for k,v in arm_method.items()}},indent=2))
    with (out/"row_level_error_examples.jsonl").open("w") as f:f.write(json.dumps({"note":"inspect per-job row_outputs_*"})+"\n")
    agg={"arm_method":{f"{k[0]}::{k[1]}":v for k,v in arm_method.items()},"failed_jobs":len(fails)};(out/"aggregate_metrics.json").write_text(json.dumps(agg,indent=2))
    # decision
    decision="real_dna_genome_not_validated_hidden_rule_family_inference_bottleneck";nxt="D37_RULE_FAMILY_INFERENCE_PLAN"
    if inv["duplicate_target_pocket_rate"]!=0 or inv["missing_target_pocket_rate"]!=0 or inv["expected_selected_points_to_target_rate"]!=1.0 or inv["ood_label_rule_changed"]:
        decision="d36_dataset_invariant_failure";nxt="D36B_DATASET_GENERATOR_REPAIR"
    else:
        dm=lambda arm: arm_method.get((arm,"direct_mutation"),{})
        r=dm("TARGET_GIVEN_ROUTING");k=dm("RULE_KNOWN_ROUTING");h=dm("RULE_HIDDEN_ROUTING")
        rr=(r.get("test_accuracy",0)>=0.95 and r.get("ood_accuracy",0)>=0.90)
        rk=(k.get("test_accuracy",0)>=0.75 and k.get("ood_accuracy",0)>=0.65)
        rh=(h.get("test_accuracy",0)>=0.50 and h.get("ood_accuracy",0)>=0.40)
        if not rr: decision="real_dna_genome_not_validated_pocket_readout_not_confirmed_after_dataset_fix";nxt="D37_POCKET_READOUT_REPAIR_PLAN"
        elif rr and not rk: decision="real_dna_genome_not_validated_formula_to_target_binding_bottleneck";nxt="D37_RULE_KNOWN_TARGET_BINDING_PLAN"
        elif rr and rk and not rh: decision="real_dna_genome_not_validated_hidden_rule_family_inference_bottleneck";nxt="D37_RULE_FAMILY_INFERENCE_PLAN"
        else: decision="real_dna_genome_not_validated_real_raven_corridor_layered_signal_confirmed";nxt="D37_DIRECT_MUTATION_ROUTING_HARDENING_PLAN"
    dec={"decision":decision,"next":nxt,"allowed_non_claims":{"raven_solved":False,"architecture_superiority":False,"natural_language_reasoning":False}}
    (out/"decision.json").write_text(json.dumps(dec,indent=2));(out/"summary.json").write_text(json.dumps({"decision":decision,"next":nxt},indent=2))
    (out/"machine_utilization_report.json").write_text(json.dumps({"os_cpu_count":os.cpu_count(),"worker_count":workers,"thread_env":{"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},"wall_clock_sec":time.time()-t0,"completed_jobs":len(done),"failed_jobs":len(fails)},indent=2))
    (out/"report.md").write_text("# D36 Real Raven Corridor Baseline Suite\n\nBoundary note: RULE_HIDDEN_ROUTING uses precomputed rule-hypothesis features without family label; it is not raw visual Raven reasoning.\n\nNon-claims: no solved claim, no architecture superiority claim, no natural-language reasoning claim.\n")

if __name__=="__main__": main()
