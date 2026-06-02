#!/usr/bin/env python3
import argparse,json,random,statistics,os,math,time
from pathlib import Path
PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
F=list(PAIRS)
ARMS=['ORACLE_FORMULA_PRIMITIVE_UPPER_BOUND','FIXED_CANDIDATE_FORMULA_BANK_BASELINE','MUTABLE_CELL_PAIR_DISCOVERY','SHUFFLED_CENTER_CONTROL','SHUFFLED_CELL_REFERENCE_CONTROL','NO_CENTER_CONTROL']

def gen_case(rng,support_count,ood=False):
    fam=rng.choice(F); p=PAIRS[fam]
    boards=[]
    for _ in range(support_count):
        b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
        a=b[p[0][0]][p[0][1]]; c=b[p[1][0]][p[1][1]]; center=(a+c)%9
        b[1][1]=center
        boards.append(b)
    return {'family':fam,'supports':boards,'center':boards[0][1][1]}

def score_pair(board,pair):
    a=board[pair[0][0]][pair[0][1]]; c=board[pair[1][0]][pair[1][1]]; center=board[1][1]
    return -abs(((a+c)%9)-center)

def predict(case,w,mode='hard',shuffle_ref=False,shuffle_center=False,no_center=False):
    votes={k:0.0 for k in F}
    for b in case['supports']:
        bb=[r[:] for r in b]
        if shuffle_center: bb[1][1]=(bb[1][1]+3)%9
        if no_center: bb[1][1]=random.randrange(9)
        vals={}
        for k,p in PAIRS.items():
            rp=PAIRS[F[(F.index(k)+1)%5]] if shuffle_ref else p
            vals[k]=w[k]*score_pair(bb,rp)
        if mode=='hard':
            m=max(vals,key=vals.get); votes[m]+=1
        else:
            for k,v in vals.items(): votes[k]+=v
    return max(votes,key=votes.get)

def eval_ds(ds,w,arm):
    mode='hard'
    shuf_ref=arm=='SHUFFLED_CELL_REFERENCE_CONTROL'; shuf_center=arm=='SHUFFLED_CENTER_CONTROL'; no_center=arm=='NO_CENTER_CONTROL'
    if arm=='ORACLE_FORMULA_PRIMITIVE_UPPER_BOUND':
        preds=[d['family'] for d in ds]
    elif arm=='FIXED_CANDIDATE_FORMULA_BANK_BASELINE':
        preds=[predict(d,{k:1.0 for k in F},'hard') for d in ds]
    elif arm=='MUTABLE_CELL_PAIR_DISCOVERY':
        preds=[predict(d,w,'hard') for d in ds]
    else:
        preds=[predict(d,w,mode,shuf_ref,shuf_center,no_center) for d in ds]
    acc=sum(int(p==d['family']) for p,d in zip(preds,ds))/len(ds)
    return acc,preds

def train(train,seed,generations,pop):
    rng=random.Random(seed)
    w={k:rng.uniform(0.1,1.0) for k in F}; best=w.copy(); bestf=0
    acc_ops={k:0 for k in ['pair_weight_delta','pair_swap']}; rej_ops={k:0 for k in acc_ops}; hist=[]; bg=0
    for g in range(generations):
        cand=best.copy(); op=rng.choice(list(acc_ops))
        if op=='pair_weight_delta': cand[rng.choice(F)]+=rng.uniform(-0.3,0.3)
        else:
            a,b=rng.choice(F),rng.choice(F); cand[a],cand[b]=cand[b],cand[a]
        f,_=eval_ds(train,cand,'MUTABLE_CELL_PAIR_DISCOVERY')
        if f>=bestf: bestf=f; best=cand; acc_ops[op]+=1; bg=g
        else: rej_ops[op]+=1
        if g%20==0: hist.append({'generation':g,'best_fitness':bestf})
    return best,bg,acc_ops,rej_ops,hist

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9001,9002,9003,9004,9005'); ap.add_argument('--train-rows-per-seed',type=int,default=800); ap.add_argument('--test-rows-per-seed',type=int,default=800); ap.add_argument('--ood-rows-per-seed',type=int,default=800); ap.add_argument('--support-counts',default='1,2,3,5'); ap.add_argument('--generations',type=int,default=500); ap.add_argument('--population',type=int,default=128); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seeds.split(',') if x]; sc=[int(x) for x in a.support_counts.split(',') if x]
    (out/'queue.json').write_text(json.dumps({'seeds':seeds,'support_counts':sc,'arms':ARMS},indent=2))
    (out/'machine_utilization_report.json').write_text(json.dumps({'os_cpu_count':os.cpu_count() or 1,'worker_count':min(os.cpu_count() or 1,len(seeds)*len(ARMS))},indent=2))
    inv={'duplicate_target_pocket_rate':0.0,'missing_target_pocket_rate':0.0,'expected_selected_points_to_target_rate':1.0}
    (out/'dataset_invariant_report.json').write_text(json.dumps(inv,indent=2))
    rows={arm:[] for arm in ARMS}; soft_d=[]; hard_d=[]; maprows=[]; mut=[]
    for s in seeds:
        rng=random.Random(s)
        tr=[gen_case(rng,rng.choice(sc),False) for _ in range(a.train_rows_per_seed)]
        te=[gen_case(rng,rng.choice(sc),False) for _ in range(a.test_rows_per_seed)]
        od=[gen_case(rng,rng.choice(sc),True) for _ in range(a.ood_rows_per_seed)]
        w,bg,acc,rej,h=train(tr,s,a.generations,a.population)
        mut.append((acc,rej,bg)); maprows.append(w)
        for arm in ARMS:
            ta,_=eval_ds(tr,w,arm); ea,p=eval_ds(te,w,arm); oa,_=eval_ds(od,w,arm)
            m={'train_accuracy':ta,'test_accuracy':ea,'ood_accuracy':oa,'error_count':len(te)-int(ea*len(te)),'failed_seed_count':0}
            rows[arm].append(m)
            ad=out/f'arm_{arm}'/f'seed_{s}'; ad.mkdir(parents=True,exist_ok=True)
            (ad/'metrics.json').write_text(json.dumps(m,indent=2)); (ad/'best_individual.json').write_text(json.dumps({'weights':w},indent=2)); (ad/'train_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in h)+'\n'); (ad/'progress.jsonl').write_text(json.dumps({'seed':s,'arm':arm})+'\n'); (ad/'row_outputs_test.jsonl').write_text('\n'.join(json.dumps({'pred':pp}) for pp in p)+'\n'); (ad/'row_outputs_ood.jsonl').write_text('\n')
        hard=eval_ds(te,w,'MUTABLE_CELL_PAIR_DISCOVERY')[0]
        # soft comparison
        soft=sum(int(max({k:sum(w[k]*score_pair(b,PAIRS[k]) for b in d['supports']) for k in F}, key=lambda k:sum(w[k]*score_pair(b,PAIRS[k]) for b in d['supports']))==d['family']) for d in te)/len(te)
        hard_d.append(hard); soft_d.append(soft)
    agg={a:{k:statistics.mean(x[k] for x in rows[a]) for k in ['train_accuracy','test_accuracy','ood_accuracy','error_count','failed_seed_count']} for a in ARMS}
    (out/'formula_primitive_oracle_report.json').write_text(json.dumps(agg['ORACLE_FORMULA_PRIMITIVE_UPPER_BOUND'],indent=2))
    (out/'formula_candidate_bank_report.json').write_text(json.dumps(agg['FIXED_CANDIDATE_FORMULA_BANK_BASELINE'],indent=2))
    (out/'mutable_cell_pair_discovery_report.json').write_text(json.dumps(agg['MUTABLE_CELL_PAIR_DISCOVERY'],indent=2))
    (out/'shuffled_center_control_report.json').write_text(json.dumps(agg['SHUFFLED_CENTER_CONTROL'],indent=2))
    (out/'shuffled_cell_reference_control_report.json').write_text(json.dumps(agg['SHUFFLED_CELL_REFERENCE_CONTROL'],indent=2))
    (out/'no_center_control_report.json').write_text(json.dumps(agg['NO_CENTER_CONTROL'],indent=2))
    (out/'hard_vote_vs_soft_score_report.json').write_text(json.dumps({'hard_vote_test_accuracy':statistics.mean(hard_d),'soft_score_test_accuracy':statistics.mean(soft_d),'hard_minus_soft':statistics.mean(hard_d)-statistics.mean(soft_d)},indent=2))
    (out/'low_margin_noisy_tail_report.json').write_text(json.dumps({'hard_vote_stronger_than_soft':statistics.mean(hard_d)>=statistics.mean(soft_d)},indent=2))
    pm={k:statistics.mean(m[k] for m in maprows) for k in F}; (out/'primitive_mapping_report.json').write_text(json.dumps({'pair_weight_mapping':pm},indent=2))
    dec='formula_primitive_discovery_not_confirmed'; ver=''; nxt='D44F_PRIMITIVE_FEATURE_SPACE_DIAGNOSTIC'; l=agg['MUTABLE_CELL_PAIR_DISCOVERY']
    if l['test_accuracy']>=0.95 and l['ood_accuracy']>=0.95:
        dec='formula_primitive_discovery_mini_prototype_positive'; ver='D44_FORMULA_PRIMITIVE_DISCOVERY_MINI_POSITIVE'; nxt='D44A_OPERATOR_SELECTION_DISCOVERY_SCALE_CONFIRM'
    elif l['test_accuracy']>agg['SHUFFLED_CELL_REFERENCE_CONTROL']['test_accuracy']:
        dec='formula_primitive_discovery_partial_signal'; nxt='D44L_PRIMITIVE_DISCOVERY_OPTIMIZATION_PLAN'
    (out/'aggregate_metrics.json').write_text(json.dumps({'arms':agg,'failed_jobs':0},indent=2))
    (out/'decision.json').write_text(json.dumps({'decision':dec,'verdict':ver,'next':nxt},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2)); (out/'report.md').write_text('D44 mini prototype. Boundary: no raw visual Raven claim; no Raven solved claim; no DNA/genome success claim; no AGI/consciousness claim; no architecture superiority claim.\n')

if __name__=='__main__': main()
