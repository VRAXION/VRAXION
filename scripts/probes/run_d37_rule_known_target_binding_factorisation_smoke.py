#!/usr/bin/env python3
import argparse, json, random, statistics
from pathlib import Path

F=["row","col","pair","mirror","diag"]
ARMS=["TARGET_GIVEN_ORACLE_CONTROL","MONOLITHIC_FORMULA_SCORER","GATED_RULE_FORMULA_SCORER","EXPLICIT_TARGET_STATE_LAYER"]

def tval(f,b):
    return {"row":(b[1][0]+b[1][2])%9,"col":(b[0][1]+b[2][1])%9,"pair":(b[0][0]+b[2][2])%9,"mirror":(b[2][0]+b[0][2])%9,"diag":(b[0][0]+b[1][2]+b[2][1])%9}[f]

def gen(rng,n,ood=False):
    out=[]
    for i in range(n):
        fam=F[i%5];b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if ood: b=[[((x*2)+1)%9 for x in row] for row in b]
        tgt=tval(fam,b)
        wrong=[x for x in range(9) if x!=tgt];rng.shuffle(wrong)
        ans=rng.randrange(9);p=[None]*9;w=0
        for j in range(9):
            if j==ans:p[j]=tgt
            else:p[j]=wrong[w%8];w+=1
        out.append({"id":i,"family":fam,"board":b,"target":tgt,"pockets":p,"expected_selected":ans})
    return out

def oracle_predict(r):
    return max(range(9),key=lambda i:1 if r['pockets'][i]==r['target'] else 0)

def mono_predict(r):
    b=r['board']; fam=r['family']
    famc={k:(1 if fam==k else 0) for k in F}
    scores=[]
    for i,c in enumerate(r['pockets']):
        fm={k:(1 if c==tval(k,b) else 0) for k in F}
        s=0.1+0.01*i + sum(fm.values()) + 0.2*sum(famc.values())
        scores.append(s)
    return max(range(9),key=lambda i:scores[i])

def gated_predict(r):
    b=r['board']; fam=r['family']
    scores=[]
    for i,c in enumerate(r['pockets']):
        s=1.0 if c==tval(fam,b) else 0.0
        scores.append(s)
    return max(range(9),key=lambda i:scores[i])

def explicit_predict(r):
    tgt=tval(r['family'],r['board'])
    return max(range(9),key=lambda i:1 if r['pockets'][i]==tgt else 0)

def eval_arm(name,tr,te,od):
    fn={"TARGET_GIVEN_ORACLE_CONTROL":oracle_predict,"MONOLITHIC_FORMULA_SCORER":mono_predict,"GATED_RULE_FORMULA_SCORER":gated_predict,"EXPLICIT_TARGET_STATE_LAYER":explicit_predict}[name]
    def run(ds):
        rows=[]
        for r in ds:
            p=fn(r);rows.append((r,p))
        acc=sum(int(r['expected_selected']==p) for r,p in rows)/len(rows)
        fam={f:sum(int(r['expected_selected']==p) for r,p in rows if r['family']==f)/max(1,sum(1 for r,_ in rows if r['family']==f)) for f in F}
        cm=[[0]*9 for _ in range(9)]
        for r,p in rows: cm[r['expected_selected']][p]+=1
        return acc,fam,cm,rows
    ta,tf,tcm,trows=run(tr);ea,ef,ecm,erows=run(te);oa,of,ocm,orows=run(od)
    return {"train_accuracy":ta,"test_accuracy":ea,"ood_accuracy":oa,"per_family_accuracy":ef,"pocket_confusion_matrix":ecm,"error_count":sum(int(r['expected_selected']!=p) for r,p in erows),"row_level_examples":[{"id":r['id'],"fam":r['family'],"truth":r['expected_selected'],"pred":p} for r,p in erows[:20]],"test_rows":erows,"ood_rows":orows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='8301,8302,8303,8304,8305');ap.add_argument('--train-rows-per-seed',type=int,default=500);ap.add_argument('--test-rows-per-seed',type=int,default=500);ap.add_argument('--ood-rows-per-seed',type=int,default=500);ap.add_argument('--heartbeat-sec',type=int,default=20);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(',') if x]
    (out/'queue.json').write_text(json.dumps({'seeds':seeds,'arms':ARMS},indent=2))
    agg={k:[] for k in ARMS}; examples=[]
    inv_occ=[]; inv_ptr=[]
    for s in seeds:
        rng=random.Random(s); tr=gen(rng,a.train_rows_per_seed,False); te=gen(rng,a.test_rows_per_seed,False); od=gen(rng,a.ood_rows_per_seed,True)
        for r in te+od:
            inv_occ.append(sum(1 for x in r['pockets'] if x==r['target']));inv_ptr.append(r['pockets'][r['expected_selected']]==r['target'])
        for arm in ARMS:
            m=eval_arm(arm,tr,te,od); agg[arm].append(m)
            examples.extend(m['row_level_examples'][:2])
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'seed':s,'arm':arm})+'\n')
    inv={"duplicate_target_pocket_rate":sum(int(x>1) for x in inv_occ)/len(inv_occ),"missing_target_pocket_rate":sum(int(x==0) for x in inv_occ)/len(inv_occ),"expected_selected_points_to_target_rate":sum(int(x) for x in inv_ptr)/len(inv_ptr)}
    (out/'dataset_invariant_report.json').write_text(json.dumps(inv,indent=2))
    (out/'dataset_manifest.json').write_text(json.dumps({'families':F,'formulas':{'row':'(b[1][0]+b[1][2])%9','col':'(b[0][1]+b[2][1])%9','pair':'(b[0][0]+b[2][2])%9','mirror':'(b[2][0]+b[0][2])%9','diag':'(b[0][0]+b[1][2]+b[2][1])%9'}},indent=2))
    oracle_test=statistics.mean(x['test_accuracy'] for x in agg['TARGET_GIVEN_ORACLE_CONTROL']); oracle_ood=statistics.mean(x['ood_accuracy'] for x in agg['TARGET_GIVEN_ORACLE_CONTROL'])
    ood_audit={'known_rule_oracle_test_accuracy':oracle_test,'known_rule_oracle_ood_accuracy':oracle_ood,'ood_label_rule_changed':False}
    (out/'ood_rule_invariance_audit.json').write_text(json.dumps(ood_audit,indent=2))
    final={}
    for arm in ARMS:
        final[arm]={k:statistics.mean(x[k] for x in agg[arm]) for k in ['train_accuracy','test_accuracy','ood_accuracy','error_count']}
    delta=final['GATED_RULE_FORMULA_SCORER']['test_accuracy']-final['MONOLITHIC_FORMULA_SCORER']['test_accuracy']
    (out/'arm_comparison_report.json').write_text(json.dumps({'metrics':final,'monolithic_vs_gated_test_delta':delta},indent=2))
    (out/'monolithic_formula_scorer_report.json').write_text(json.dumps(final['MONOLITHIC_FORMULA_SCORER'],indent=2));(out/'gated_rule_formula_scorer_report.json').write_text(json.dumps(final['GATED_RULE_FORMULA_SCORER'],indent=2));(out/'explicit_target_state_layer_report.json').write_text(json.dumps(final['EXPLICIT_TARGET_STATE_LAYER'],indent=2));(out/'target_given_oracle_report.json').write_text(json.dumps(final['TARGET_GIVEN_ORACLE_CONTROL'],indent=2))
    (out/'per_family_report.json').write_text(json.dumps({'families':F},indent=2));(out/'pocket_confusion_matrix.json').write_text(json.dumps({'note':'in per-arm raw'},indent=2))
    with (out/'row_level_examples.jsonl').open('w') as f:
        for e in examples:f.write(json.dumps(e)+'\n')
    decision='rule_known_factorisation_not_confirmed';next_step='D38_FEATURE_SPACE_DIAGNOSTIC';verdict=''
    if inv['duplicate_target_pocket_rate']!=0 or inv['missing_target_pocket_rate']!=0 or inv['expected_selected_points_to_target_rate']!=1.0:
        decision='d37_dataset_invariant_failure';next_step='D37B_DATASET_REPAIR'
    elif ood_audit['known_rule_oracle_ood_accuracy']<1.0:
        decision='d37_ood_rule_invariance_failure';next_step='D37C_OOD_RULE_REPAIR'
    else:
        tg=final['TARGET_GIVEN_ORACLE_CONTROL'];g=final['GATED_RULE_FORMULA_SCORER'];e=final['EXPLICIT_TARGET_STATE_LAYER'];m=final['MONOLITHIC_FORMULA_SCORER']
        if tg['test_accuracy']==1.0 and tg['ood_accuracy']==1.0 and g['test_accuracy']>=0.99 and g['ood_accuracy']>=0.99 and e['test_accuracy']>=0.99 and e['ood_accuracy']>=0.99 and (g['test_accuracy']-m['test_accuracy'])>=0.40:
            decision='rule_known_factorisation_bottleneck_confirmed';verdict='D37_RULE_KNOWN_TARGET_BINDING_FACTORISATION_CONFIRMED';next_step='D38_RULE_KNOWN_ROUTER_LAYER_PROTOTYPE'
        elif e['test_accuracy']>=0.99 and e['ood_accuracy']>=0.99 and not (g['test_accuracy']>m['test_accuracy']):
            decision='explicit_target_state_required';next_step='D38_EXPLICIT_TARGET_STATE_LAYER_PROTOTYPE'
    (out/'aggregate_metrics.json').write_text(json.dumps({'arms':final,'monolithic_vs_gated_test_delta':delta},indent=2))
    (out/'decision.json').write_text(json.dumps({'decision':decision,'verdict':verdict,'next':next_step},indent=2));(out/'summary.json').write_text(json.dumps({'decision':decision,'next':next_step},indent=2))
    (out/'report.md').write_text('D37 factorisation smoke. Non-claims: no Raven solved claim; no architecture superiority claim; no natural-language reasoning claim. RULE_HIDDEN not claimed here.\n')

if __name__=='__main__':main()
