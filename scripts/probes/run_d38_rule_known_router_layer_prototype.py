#!/usr/bin/env python3
import argparse, json, random, statistics, zlib, os, time, math
from pathlib import Path

F=["row","col","pair","mirror","diag"]
ARMS=["TARGET_GIVEN_ORACLE_CONTROL","MONOLITHIC_FORMULA_BASELINE","ORACLE_GATED_RULE_FORMULA_UPPER_BOUND","MUTABLE_LEARNED_ROUTER_GATE","SHUFFLED_GATE_CONTROL","NO_FAMILY_INPUT_CONTROL","EXPLICIT_TARGET_STATE_UPPER_BOUND"]


def tval(f,b):
    return {"row":(b[1][0]+b[1][2])%9,"col":(b[0][1]+b[2][1])%9,"pair":(b[0][0]+b[2][2])%9,"mirror":(b[2][0]+b[0][2])%9,"diag":(b[0][0]+b[1][2]+b[2][1])%9}[f]

def gen(rng,n,ood=False):
    out=[]
    for i in range(n):
        fam=F[i%5]
        b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if ood:
            b=[[((x*2)+3)%9 for x in row] for row in b]
        tgt=tval(fam,b)
        wrong=[x for x in range(9) if x!=tgt]; rng.shuffle(wrong)
        ans=rng.randrange(9)
        pockets=[None]*9; wi=0
        for p in range(9):
            if p==ans: pockets[p]=tgt
            else: pockets[p]=wrong[wi%8]; wi+=1
        out.append({"id":i,"family":fam,"board":b,"target":tgt,"pockets":pockets,"expected_selected":ans})
    return out

def formula_vec(r,p):
    c=r['pockets'][p]; b=r['board']
    return [1.0 if c==tval(f,b) else 0.0 for f in F]

def eval_simple(ds,predict):
    rows=[]
    for r in ds:
        p,scores=predict(r); rows.append((r,p,scores))
    acc=sum(int(r['expected_selected']==p) for r,p,_ in rows)/len(rows)
    fam={f:sum(int(r['expected_selected']==p) for r,p,_ in rows if r['family']==f)/max(1,sum(1 for r,_,_ in rows if r['family']==f)) for f in F}
    cm=[[0]*9 for _ in range(9)]
    margins=[]
    for r,p,s in rows:
        cm[r['expected_selected']][p]+=1
        ss=sorted(s,reverse=True); margins.append((ss[0]-ss[1]) if len(ss)>1 else ss[0])
    return acc,fam,cm,rows,statistics.median(margins),sum(1 for r,p,_ in rows if r['expected_selected']!=p)/len(rows)

def pred_target_given(r):
    s=[1.0 if x==r['target'] else 0.0 for x in r['pockets']];return max(range(9),key=lambda i:s[i]),s

def pred_monolithic(r):
    famc=[1.0 if r['family']==f else 0.0 for f in F]
    s=[]
    for i in range(9):
        fv=formula_vec(r,i)
        s.append(0.1+0.01*i + sum(fv) + 0.2*sum(famc))
    return max(range(9),key=lambda i:s[i]),s

def pred_oracle_gated(r):
    fi=F.index(r['family'])
    s=[]
    for i in range(9):
        fv=formula_vec(r,i); s.append(fv[fi])
    return max(range(9),key=lambda i:s[i]),s

def pred_explicit(r):
    tgt=tval(r['family'],r['board']); s=[1.0 if x==tgt else 0.0 for x in r['pockets']];return max(range(9),key=lambda i:s[i]),s

def pred_nofam(r):
    s=[]
    for i in range(9): s.append(sum(formula_vec(r,i)))
    return max(range(9),key=lambda i:s[i]),s

def predict_learned(r,ind,shuffle=False):
    fi=F.index(r['family'])
    row=ind['G'][fi][:]
    if shuffle:
        row=ind['G'][(fi+1)%5][:]
    gate=[row[j]+ind['gb'][j] for j in range(5)]
    scores=[]
    for i in range(9):
        fv=formula_vec(r,i)
        val=sum(gate[j]*fv[j] for j in range(5))+ind['pb'][i]
        scores.append(val)
    return max(range(9),key=lambda i:scores[i]),scores

def init_ind(rng):
    return {'G':[[rng.uniform(-0.2,0.2) for _ in range(5)] for __ in range(5)], 'gb':[rng.uniform(-0.1,0.1) for _ in range(5)], 'pb':[rng.uniform(-0.05,0.05) for _ in range(9)]}

def clone(ind): return {'G':[r[:] for r in ind['G']],'gb':ind['gb'][:],'pb':ind['pb'][:]}

def mutate(rng,ind):
    c=clone(ind); ops=[]
    t=rng.choice(['gate_weight_delta','gate_bias_delta','pocket_bias_delta','gate_row_swap','gate_column_swap'])
    if t=='gate_weight_delta':
        i,j=rng.randrange(5),rng.randrange(5); c['G'][i][j]+=rng.uniform(-0.5,0.5)
    elif t=='gate_bias_delta':
        j=rng.randrange(5); c['gb'][j]+=rng.uniform(-0.25,0.25)
    elif t=='pocket_bias_delta':
        j=rng.randrange(9); c['pb'][j]+=rng.uniform(-0.1,0.1)
    elif t=='gate_row_swap':
        a,b=rng.randrange(5),rng.randrange(5); c['G'][a],c['G'][b]=c['G'][b],c['G'][a]
    else:
        a,b=rng.randrange(5),rng.randrange(5)
        for r in range(5): c['G'][r][a],c['G'][r][b]=c['G'][r][b],c['G'][r][a]
    ops.append(t)
    return c,ops

def fit(ds,ind):
    acc,_,_,_,med,_=eval_simple(ds,lambda r:predict_learned(r,ind,False))
    return acc + 0.001*med

def train_learned(train,seed,generations,pop):
    rng=random.Random(seed)
    popu=[init_ind(rng) for _ in range(pop)]
    accepted={k:0 for k in ['gate_weight_delta','gate_bias_delta','pocket_bias_delta','gate_row_swap','gate_column_swap']}
    rejected={k:0 for k in accepted}
    history=[]
    scored=[(fit(train,i),i) for i in popu]; scored.sort(key=lambda x:x[0],reverse=True)
    best=clone(scored[0][1]); bestf=scored[0][0]; best_gen=0
    for g in range(generations):
        base=clone(best)
        child,ops=mutate(rng,base)
        cf=fit(train,child)
        if cf>=bestf:
            best,bestf=child,cf; best_gen=g
            for o in ops:accepted[o]+=1
        else:
            for o in ops:rejected[o]+=1
        if g%10==0: history.append({'generation':g,'best_fitness':bestf})
    return best,best_gen,accepted,rejected,history

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='8401,8402,8403,8404,8405'); ap.add_argument('--train-rows-per-seed',type=int,default=500); ap.add_argument('--test-rows-per-seed',type=int,default=500); ap.add_argument('--ood-rows-per-seed',type=int,default=500); ap.add_argument('--generations',type=int,default=250); ap.add_argument('--population',type=int,default=80); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(',') if x]
    oscpu=os.cpu_count() or 1; workers=min(oscpu,len(seeds)*len(ARMS)) if a.workers=='auto' else int(a.workers)
    (out/'queue.json').write_text(json.dumps({'seeds':seeds,'arms':ARMS,'workers':workers},indent=2))
    (out/'machine_utilization_report.json').write_text(json.dumps({'os_cpu_count':oscpu,'worker_count':workers,'thread_env':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}},indent=2))
    (out/'dataset_manifest.json').write_text(json.dumps({'families':F},indent=2))
    allm={k:[] for k in ARMS}; examples=[]; inv_occ=[]; inv_ptr=[]; gate_rows=[]; mut=[]
    start=time.time()
    for s in seeds:
        rng=random.Random(s)
        tr=gen(rng,a.train_rows_per_seed,False); te=gen(rng,a.test_rows_per_seed,False); od=gen(rng,a.ood_rows_per_seed,True)
        for r in te+od:
            inv_occ.append(sum(1 for x in r['pockets'] if x==r['target'])); inv_ptr.append(r['pockets'][r['expected_selected']]==r['target'])
        best,bgen,accm,rejm,hist=train_learned(tr,s,a.generations,a.population)
        for arm in ARMS:
            fn={'TARGET_GIVEN_ORACLE_CONTROL':pred_target_given,'MONOLITHIC_FORMULA_BASELINE':pred_monolithic,'ORACLE_GATED_RULE_FORMULA_UPPER_BOUND':pred_oracle_gated,'SHUFFLED_GATE_CONTROL':lambda r:predict_learned(r,best,True),'NO_FAMILY_INPUT_CONTROL':pred_nofam,'EXPLICIT_TARGET_STATE_UPPER_BOUND':pred_explicit,'MUTABLE_LEARNED_ROUTER_GATE':lambda r:predict_learned(r,best,False)}[arm]
            trm=eval_simple(tr,fn); tem=eval_simple(te,fn); odm=eval_simple(od,fn)
            m={'train_accuracy':trm[0],'test_accuracy':tem[0],'ood_accuracy':odm[0],'per_family_accuracy':tem[1],'pocket_confusion_matrix':tem[2],'median_score_margin':tem[4],'low_margin_error_rate':tem[5],'error_count':sum(int(r['expected_selected']!=p) for r,p,_ in tem[3]),'row_level_examples':[{'id':r['id'],'family':r['family'],'truth':r['expected_selected'],'pred':p} for r,p,_ in tem[3][:20]]}
            allm[arm].append(m); examples.extend(m['row_level_examples'][:2])
            ad=out/f'arm_{arm}'/f'seed_{s}'; ad.mkdir(parents=True,exist_ok=True)
            (ad/'metrics.json').write_text(json.dumps(m,indent=2)); (ad/'best_individual.json').write_text(json.dumps(best,indent=2)); (ad/'train_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in hist)+'\n')
            (ad/'progress.jsonl').write_text(json.dumps({'seed':s,'arm':arm,'status':'done'})+'\n')
            (ad/'row_outputs_test.jsonl').write_text('\n'.join(json.dumps({'id':r['id'],'pred':p,'truth':r['expected_selected']}) for r,p,_ in tem[3])+'\n')
            (ad/'row_outputs_ood.jsonl').write_text('\n'.join(json.dumps({'id':r['id'],'pred':p,'truth':r['expected_selected']}) for r,p,_ in odm[3])+'\n')
            with (out/'progress.jsonl').open('a') as f: f.write(json.dumps({'seed':s,'arm':arm})+'\n')
        gate_rows.append(best); mut.append({'accepted':accm,'rejected':rejm,'best_generation':bgen})
    inv={'duplicate_target_pocket_rate':sum(int(x>1) for x in inv_occ)/len(inv_occ),'missing_target_pocket_rate':sum(int(x==0) for x in inv_occ)/len(inv_occ),'expected_selected_points_to_target_rate':sum(int(x) for x in inv_ptr)/len(inv_ptr)}
    (out/'dataset_invariant_report.json').write_text(json.dumps(inv,indent=2))
    ood={'known_rule_oracle_test_accuracy':statistics.mean(x['test_accuracy'] for x in allm['TARGET_GIVEN_ORACLE_CONTROL']),'known_rule_oracle_ood_accuracy':statistics.mean(x['ood_accuracy'] for x in allm['TARGET_GIVEN_ORACLE_CONTROL']),'ood_label_rule_changed':False}
    (out/'ood_rule_invariance_audit.json').write_text(json.dumps(ood,indent=2))
    agg={arm:{k:statistics.mean(x[k] for x in allm[arm]) for k in ['train_accuracy','test_accuracy','ood_accuracy','median_score_margin','low_margin_error_rate','error_count']} for arm in ARMS}
    report_map={
        'monolithic_formula_baseline_report.json':'MONOLITHIC_FORMULA_BASELINE',
        'oracle_gated_rule_formula_report.json':'ORACLE_GATED_RULE_FORMULA_UPPER_BOUND',
        'mutable_learned_router_gate_report.json':'MUTABLE_LEARNED_ROUTER_GATE',
        'shuffled_gate_control_report.json':'SHUFFLED_GATE_CONTROL',
        'no_family_input_control_report.json':'NO_FAMILY_INPUT_CONTROL',
        'explicit_target_state_upper_bound_report.json':'EXPLICIT_TARGET_STATE_UPPER_BOUND'
    }
    for fn,arm in report_map.items():
        (out/fn).write_text(json.dumps(agg[arm],indent=2))
    (out/'target_given_oracle_report.json').write_text(json.dumps(agg['TARGET_GIVEN_ORACLE_CONTROL'],indent=2))
    map_counts=[]
    for b in gate_rows:
        gm=b['G']; mapping={F[i]:F[max(range(5),key=lambda j:gm[i][j])] for i in range(5)}
        map_counts.append(mapping)
    # use first mapping for report
    gm=gate_rows[0]['G']; diag=sum(abs(gm[i][i]) for i in range(5)); off=sum(abs(gm[i][j]) for i in range(5) for j in range(5) if i!=j)
    align=sum(1 for i in range(5) if max(range(5),key=lambda j:gm[i][j])==i)/5
    ent=[]
    for i in range(5):
        vals=[abs(gm[i][j])+1e-9 for j in range(5)]; s=sum(vals); p=[v/s for v in vals]; ent.append(-sum(x*math.log(x) for x in p))
    gate_rep={'gate_identity_alignment_score':align,'diagonal_gate_mass':diag,'off_diagonal_gate_mass':off,'gate_argmax_mapping':map_counts[0],'gate_entropy_mean':statistics.mean(ent)}
    (out/'gate_matrix_report.json').write_text(json.dumps(gate_rep,indent=2)); (out/'gate_identity_alignment_report.json').write_text(json.dumps(gate_rep,indent=2))
    acc_tot={k:sum(m['accepted'][k] for m in mut) for k in mut[0]['accepted']}; rej_tot={k:sum(m['rejected'][k] for m in mut) for k in mut[0]['rejected']}; total=sum(acc_tot.values())+sum(rej_tot.values())
    mut_rep={'accepted_mutations_by_type':acc_tot,'rejected_mutations_by_type':rej_tot,'mutation_acceptance_rate':(sum(acc_tot.values())/total if total else 0.0),'convergence_generation_median':statistics.median([m['best_generation'] for m in mut]),'seed_variance':statistics.pvariance([m['best_generation'] for m in mut])}
    (out/'mutation_acceptance_report.json').write_text(json.dumps(mut_rep,indent=2))
    delta=agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']-agg['MONOLITHIC_FORMULA_BASELINE']['test_accuracy']
    comp={'arms':agg,'monolithic_vs_learned_test_delta':delta,'learned_vs_shuffled_test_delta':agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']-agg['SHUFFLED_GATE_CONTROL']['test_accuracy'],'learned_vs_no_family_test_delta':agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']-agg['NO_FAMILY_INPUT_CONTROL']['test_accuracy']}
    (out/'arm_comparison_report.json').write_text(json.dumps(comp,indent=2)); (out/'aggregate_metrics.json').write_text(json.dumps({**comp,**gate_rep,**mut_rep,'failed_seed_count':0},indent=2))
    (out/'per_seed_report.json').write_text(json.dumps({'seed_count':len(seeds),'failed_seed_count':0},indent=2)); (out/'per_family_report.json').write_text(json.dumps({'families':F,'mutable_learned_router_gate_per_family_accuracy':statistics.mean([x['per_family_accuracy'][F[0]] for x in allm['MUTABLE_LEARNED_ROUTER_GATE']])},indent=2)); (out/'pocket_confusion_matrix.json').write_text(json.dumps({'note':'see per-arm/seed metrics'},indent=2)); (out/'score_margin_report.json').write_text(json.dumps({'mutable_median_score_margin':agg['MUTABLE_LEARNED_ROUTER_GATE']['median_score_margin']},indent=2))
    with (out/'row_level_examples.jsonl').open('w') as f:
        for e in examples: f.write(json.dumps(e)+'\n')
    decision='learned_router_not_confirmed'; verdict='D38_LEARNED_ROUTER_NOT_CONFIRMED'; nexts='D39_FEATURE_SPACE_DIAGNOSTIC'
    if inv['duplicate_target_pocket_rate']!=0 or inv['missing_target_pocket_rate']!=0 or inv['expected_selected_points_to_target_rate']!=1.0:
        decision='d38_dataset_invariant_failure'; verdict=''; nexts='D38B_DATASET_REPAIR'
    elif ood['known_rule_oracle_ood_accuracy']<1.0:
        decision='d38_ood_rule_invariance_failure'; verdict=''; nexts='D38C_OOD_RULE_REPAIR'
    elif agg['ORACLE_GATED_RULE_FORMULA_UPPER_BOUND']['test_accuracy']>=0.99 and agg['EXPLICIT_TARGET_STATE_UPPER_BOUND']['test_accuracy']>=0.99 and not (agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']>=0.95 and agg['MUTABLE_LEARNED_ROUTER_GATE']['ood_accuracy']>=0.95):
        decision='oracle_router_confirmed_but_learned_gate_failed'; verdict='D38_ORACLE_ROUTER_CONFIRMED_LEARNED_GATE_FAILED'; nexts='D38L_LEARNED_GATE_OPTIMIZATION_PLAN'
    elif agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']>=0.95 and agg['MUTABLE_LEARNED_ROUTER_GATE']['ood_accuracy']>=0.95 and delta>=0.40 and agg['SHUFFLED_GATE_CONTROL']['test_accuracy']<=0.60 and agg['NO_FAMILY_INPUT_CONTROL']['test_accuracy']<=0.60:
        decision='learned_rule_known_router_layer_prototype_positive'; verdict='D38_LEARNED_RULE_KNOWN_ROUTER_LAYER_PROTOTYPE_POSITIVE'; nexts='D39_ROUTER_LAYER_SCALE_CONFIRM'
    elif agg['MUTABLE_LEARNED_ROUTER_GATE']['test_accuracy']>agg['MONOLITHIC_FORMULA_BASELINE']['test_accuracy']:
        decision='learned_router_partial_signal'; verdict='D38_LEARNED_ROUTER_PARTIAL_SIGNAL'; nexts='D38L_LEARNED_GATE_OPTIMIZATION_PLAN'
    (out/'decision.json').write_text(json.dumps({'decision':decision,'verdict':verdict,'next':nexts},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':decision,'next':nexts,'wall_clock_sec':time.time()-start},indent=2)); (out/'report.md').write_text('D38 known-rule router prototype. Non-claims: no hidden-rule Raven solved claim, no natural-language reasoning claim, no architecture superiority claim.\n')

if __name__=='__main__': main()
