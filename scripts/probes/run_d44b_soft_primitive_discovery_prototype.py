#!/usr/bin/env python3
import argparse,json,random,statistics
from pathlib import Path
from collections import Counter,defaultdict

PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
F=list(PAIRS)
ARMS=['TRUE_LABEL_ECHO_REFERENCE_ONLY','FAIR_IDENTIFIABILITY_UPPER_BOUND','FIXED_HARD_INSERTION_ORDER_BASELINE','FIXED_HARD_RANDOM_TIE_BASELINE','FIXED_SOFT_SCORE_BASELINE','COLLISION_AWARE_SOFT_SCORE','SOFT_PREFILTER_ELIMINATION','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID','ORDER_SHUFFLE_CONTROL','SHUFFLED_CENTER_CONTROL','SHUFFLED_CELL_REFERENCE_CONTROL','NO_CENTER_CONTROL']

def gen_case(rng,support_count,ood,row_id):
    fam=rng.choice(F); p=PAIRS[fam]; su=[]
    for _ in range(support_count):
        b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
        b[1][1]=(b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9
        su.append(b)
    return {'row_id':row_id,'truth_family':fam,'supports':su,'support_count':support_count}

def score_pair(b,p): return -abs(((b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9)-b[1][1])

def normalize(sc):
    vals=list(sc.values()); mn,mx=min(vals),max(vals)
    if mx==mn: return {k:0.0 for k in sc}
    return {k:(v-mn)/(mx-mn) for k,v in sc.items()}

def predict(row,arm,w,seed):
    rng=random.Random(seed*100000+row['row_id'])
    if arm=='TRUE_LABEL_ECHO_REFERENCE_ONLY':
        pred=row['truth_family']; return pred,pred,pred,{}, {}, [],0,[],[],0.0,pred,pred,[],[pred],False
    sh_center=arm=='SHUFFLED_CENTER_CONTROL'; sh_ref=arm=='SHUFFLED_CELL_REFERENCE_CONTROL'; no_center=arm=='NO_CENTER_CONTROL'; order_shuffle=arm=='ORDER_SHUFFLE_CONTROL'
    wk={k:1.0 for k in F} if arm.startswith('FIXED_') or arm in ['COLLISION_AWARE_SOFT_SCORE','SOFT_PREFILTER_ELIMINATION','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID','ORDER_SHUFFLE_CONTROL','FAIR_IDENTIFIABILITY_UPPER_BOUND'] else w
    supports_winners=[]; vote=Counter(); soft=defaultdict(float); elim=[]
    for b in row['supports']:
        bb=[r[:] for r in b]
        if sh_center: bb[1][1]=(bb[1][1]+3)%9
        if no_center: bb[1][1]=rng.randrange(9)
        keys=F[:]
        if order_shuffle: rng.shuffle(keys)
        sc={}
        for k in keys:
            p=PAIRS[F[(F.index(k)+1)%5]] if sh_ref else PAIRS[k]
            sc[k]=wk[k]*score_pair(bb,p)
        m=max(sc.values()); tied=[k for k,v in sc.items() if v==m]
        ins=tied[0]; rnd=rng.choice(tied)
        use=rnd if arm=='FIXED_HARD_RANDOM_TIE_BASELINE' else ins
        vote[use]+=1; supports_winners.append(use)
        ns=normalize(sc)
        for k,v in (ns.items() if arm in ['COLLISION_AWARE_SOFT_SCORE','SOFT_PREFILTER_ELIMINATION','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID'] else sc.items()): soft[k]+=v
    candidate_scores=dict(soft); normalized_scores=normalize(candidate_scores)
    sortedc=sorted(candidate_scores.items(),key=lambda x:x[1],reverse=True)
    margin=(sortedc[0][1]-sortedc[1][1]) if len(sortedc)>1 else sortedc[0][1]
    tied=[k for k,v in candidate_scores.items() if v==sortedc[0][1]]
    exact=[k for k,v in candidate_scores.items() if abs(v-sortedc[0][1])<1e-12]
    collision_count=max(0,len(tied)-1)
    hard_pred=max(vote,key=vote.get) if vote else sortedc[0][0]
    soft_pred=sortedc[0][0]
    pred=soft_pred
    ambiguous=False
    if arm=='FAIR_IDENTIFIABILITY_UPPER_BOUND':
        if collision_count>0: ambiguous=True
    elif arm=='FIXED_HARD_INSERTION_ORDER_BASELINE' or arm=='FIXED_HARD_RANDOM_TIE_BASELINE': pred=hard_pred
    elif arm=='SOFT_PREFILTER_ELIMINATION':
        thr=sortedc[0][1]-0.2; surv=[k for k,v in sortedc if v>=thr]; elim=[k for k,v in sortedc if v<thr]; pred=surv[0]
    elif arm=='ITERATIVE_ELIMINATION_PAIRWISE':
        thr=sortedc[0][1]-0.15; surv=[k for k,v in sortedc if v>=thr]; elim=[k for k,v in sortedc if v<thr]
        if len(surv)>1:
            duel=Counter()
            for b in row['supports']:
                best=max(surv,key=lambda k:score_pair(b,PAIRS[k])); duel[best]+=1
            pred=max(duel,key=duel.get)
        else: pred=surv[0]
    elif arm=='SOFT_THEN_HARD_HYBRID':
        thr=sortedc[0][1]-0.2; surv=[k for k,v in sortedc if v>=thr]; elim=[k for k,v in sortedc if v<thr]
        hv=Counter([x for x in supports_winners if x in surv]); pred=max(hv,key=hv.get) if hv else surv[0]
    return pred,hard_pred,soft_pred,candidate_scores,normalized_scores,supports_winners,collision_count,exact,tied,margin,tied[0],random.Random(seed+row['row_id']).choice(tied),elim,[k for k,_ in sortedc if k not in elim],ambiguous

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9101,9102,9103,9104,9105'); ap.add_argument('--train-rows-per-seed',type=int,default=1000); ap.add_argument('--test-rows-per-seed',type=int,default=1000); ap.add_argument('--ood-rows-per-seed',type=int,default=1000); ap.add_argument('--support-counts',default='1,2,3,5'); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    d44a_dec='missing'
    if Path('target/pilot_wave/d44a_formula_primitive_discovery_diagnostic/smoke/decision.json').exists():
        d44a_dec=json.loads(Path('target/pilot_wave/d44a_formula_primitive_discovery_diagnostic/smoke/decision.json').read_text())
    (out/'d44a_upstream_manifest.json').write_text(json.dumps({'d44a_result_doc_exists':Path('docs/research/D44A_FORMULA_PRIMITIVE_DISCOVERY_DIAGNOSTIC_RESULT.md').exists(),'d44a_runner_exists':Path('scripts/probes/run_d44a_formula_primitive_discovery_diagnostic.py').exists(),'d44a_decision':d44a_dec},indent=2))
    (out/'dataset_manifest.json').write_text(json.dumps({'families':F,'pairs':PAIRS,'support_counts':a.support_counts},indent=2))
    seeds=[int(x) for x in a.seeds.split(',') if x]; sc=[int(x) for x in a.support_counts.split(',') if x]
    rows={arm:{'train':[],'test':[],'ood':[]} for arm in ARMS}
    for sd in seeds:
        rng=random.Random(sd); w={k:rng.uniform(0.1,1.0) for k in F}
        ds={'train':[gen_case(rng,rng.choice(sc),False,i) for i in range(a.train_rows_per_seed)],'test':[gen_case(rng,rng.choice(sc),False,i) for i in range(a.test_rows_per_seed)],'ood':[gen_case(rng,rng.choice(sc),True,i) for i in range(a.ood_rows_per_seed)]}
        for arm in ARMS:
            for split in ['train','test','ood']:
                for r in ds[split]:
                    pred,hard,soft,csc,nsc,psw,cc,exact,tied,margin,ins,rnd,elim,surv,amb=predict(r,arm,w,sd)
                    rows[arm][split].append({'row_id':r['row_id'],'seed':sd,'split':split,'truth_family':r['truth_family'],'pred_family':pred,'support_count':r['support_count'],'arm':arm,'hard_pred':hard,'soft_pred':soft,'candidate_scores':csc,'normalized_scores':nsc,'per_support_winners':psw,'collision_count':cc,'exact_match_candidates':exact,'tied_candidates':tied,'margin_to_runner_up':margin,'insertion_order_winner':ins,'random_tie_winner':rnd,'eliminated_candidates_by_round':[elim],'final_survivors':surv,'correct':int(pred==r['truth_family']),'error_type':'ambiguous' if amb else ('ok' if pred==r['truth_family'] else 'misclass'),'ambiguous':amb})
    for split in ['train','test','ood']:
        with (out/f'row_outputs_{split}.jsonl').open('w') as f:
            for arm in ARMS:
                for r in rows[arm][split]: f.write(json.dumps(r)+'\n')
    def acc(rs): return sum(r['correct'] for r in rs)/len(rs)
    agg={'arms':{},'failed_jobs':0}
    per_family={}; conf={}; hvsf={}; hvss={str(s):{} for s in sc}; hvsc={}
    for arm in ARMS:
        t=rows[arm]['test']; o=rows[arm]['ood']; tr=rows[arm]['train']
        agg['arms'][arm]={'train_accuracy':acc(tr),'test_accuracy':acc(t),'OOD_accuracy':acc(o),'error_count':sum(1-r['correct'] for r in t),'failed_seed_count':0}
        per_family[arm]={f:acc([r for r in t if r['truth_family']==f]) for f in F}
        cm={f:{g:0 for g in F} for f in F}
        for r in t: cm[r['truth_family']][r['pred_family']]+=1
        conf[arm]=cm
    for f in F: hvsf[f]={'hard':per_family['FIXED_HARD_INSERTION_ORDER_BASELINE'][f],'soft':per_family['FIXED_SOFT_SCORE_BASELINE'][f]}
    for s in sc:
        for arm in ['FIXED_HARD_INSERTION_ORDER_BASELINE','FIXED_SOFT_SCORE_BASELINE','COLLISION_AWARE_SOFT_SCORE','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID']:
            hvss[str(s)][arm]=acc([r for r in rows[arm]['test'] if r['support_count']==s])
    for c in [0,1,2,3,4]:
        hvsc[str(c)]={}
        for arm in ['FIXED_HARD_INSERTION_ORDER_BASELINE','FIXED_SOFT_SCORE_BASELINE']:
            xs=[r for r in rows[arm]['test'] if r['collision_count']==c]
            if xs: hvsc[str(c)][arm]=acc(xs)
    coll_rows=rows['FIXED_HARD_INSERTION_ORDER_BASELINE']['test']
    collision={'exact_collision_error_rate':sum(1 for r in coll_rows if r['collision_count']>0 and r['correct']==0)/len(coll_rows),'near_collision_error_rate':sum(1 for r in coll_rows if r['margin_to_runner_up']<=0.05 and r['correct']==0)/len(coll_rows)}
    tie={'candidate_order_sensitivity':sum(int(a['pred_family']!=b['pred_family']) for a,b in zip(rows['ORDER_SHUFFLE_CONTROL']['test'],rows['FIXED_HARD_INSERTION_ORDER_BASELINE']['test']))/len(coll_rows)}
    fair=rows['FAIR_IDENTIFIABILITY_UPPER_BOUND']['test']; fair_upper={'fair_upper_bound_accuracy':acc(fair),'identifiable_rate':sum(int(not r['ambiguous']) for r in fair)/len(fair),'ambiguous_rate':sum(int(r['ambiguous']) for r in fair)/len(fair)}
    for s in sc:
        xs=[r for r in fair if r['support_count']==s]; fair_upper[f'support_count_{s}_upper_bound']=acc(xs)
    iter_test=rows['ITERATIVE_ELIMINATION_PAIRWISE']['test']; softb=agg['arms']['FIXED_SOFT_SCORE_BASELINE']['test_accuracy']; hardb=agg['arms']['FIXED_HARD_INSERTION_ORDER_BASELINE']['test_accuracy']
    iter_rep={'test_accuracy':acc(iter_test),'OOD_accuracy':agg['arms']['ITERATIVE_ELIMINATION_PAIRWISE']['OOD_accuracy'],'eliminated_correct_candidate_rate':sum(int(r['truth_family'] in r['eliminated_candidates_by_round'][0]) for r in iter_test)/len(iter_test),'average_candidates_eliminated':statistics.mean(len(r['eliminated_candidates_by_round'][0]) for r in iter_test),'average_final_survivors':statistics.mean(len(r['final_survivors']) for r in iter_test),'abstain_or_ambiguous_rate':sum(int(r['ambiguous']) for r in iter_test)/len(iter_test),'pairwise_win_matrix':conf['ITERATIVE_ELIMINATION_PAIRWISE'],'fair_upper_bound_gap':fair_upper['fair_upper_bound_accuracy']-acc(iter_test),'soft_baseline_delta':acc(iter_test)-softb,'hard_baseline_delta':acc(iter_test)-hardb}
    softhy={'test_accuracy':agg['arms']['SOFT_THEN_HARD_HYBRID']['test_accuracy'],'OOD_accuracy':agg['arms']['SOFT_THEN_HARD_HYBRID']['OOD_accuracy']}
    order={'order_shuffle_test_accuracy':agg['arms']['ORDER_SHUFFLE_CONTROL']['test_accuracy'],'order_sensitivity':tie['candidate_order_sensitivity']}
    # write reports
    (out/'per_family_accuracy_report.json').write_text(json.dumps(per_family,indent=2)); (out/'confusion_matrix_report.json').write_text(json.dumps(conf,indent=2));
    (out/'collision_report.json').write_text(json.dumps(collision,indent=2));
    (out/'collision_by_family_report.json').write_text(json.dumps({f:sum(1 for r in coll_rows if r['truth_family']==f and r['collision_count']>0)/max(1,sum(1 for r in coll_rows if r['truth_family']==f)) for f in F},indent=2));
    (out/'collision_by_support_count_report.json').write_text(json.dumps({str(s):sum(1 for r in coll_rows if r['support_count']==s and r['collision_count']>0)/max(1,sum(1 for r in coll_rows if r['support_count']==s)) for s in sc},indent=2));
    (out/'hard_vs_soft_by_family_report.json').write_text(json.dumps(hvsf,indent=2)); (out/'hard_vs_soft_by_support_count_report.json').write_text(json.dumps(hvss,indent=2)); (out/'hard_vs_soft_by_collision_count_report.json').write_text(json.dumps(hvsc,indent=2)); (out/'tie_bias_report.json').write_text(json.dumps(tie,indent=2));
    (out/'fair_identifiability_upper_bound_report.json').write_text(json.dumps(fair_upper,indent=2)); (out/'order_shuffle_control_report.json').write_text(json.dumps(order,indent=2));
    (out/'soft_prefilter_elimination_report.json').write_text(json.dumps({'test_accuracy':agg['arms']['SOFT_PREFILTER_ELIMINATION']['test_accuracy']},indent=2));
    (out/'iterative_elimination_pairwise_report.json').write_text(json.dumps(iter_rep,indent=2));
    (out/'soft_then_hard_hybrid_report.json').write_text(json.dumps(softhy,indent=2));
    (out/'margin_strata_report.json').write_text(json.dumps({'low_margin_rate':sum(int(r['margin_to_runner_up']<=0.05) for r in coll_rows)/len(coll_rows)},indent=2));
    (out/'error_taxonomy_report.json').write_text(json.dumps({'misclass_rate':sum(int(r['error_type']=='misclass') for r in coll_rows)/len(coll_rows),'ambiguous_rate':sum(int(r['error_type']=='ambiguous') for r in coll_rows)/len(coll_rows)},indent=2));
    with (out/'row_level_error_examples.jsonl').open('w') as f:
        for r in coll_rows[:200]:
            if not r['correct']: f.write(json.dumps(r)+'\n')
    # decision
    dec='d44b_no_improvement_over_soft_baseline'; ver='D44B_NO_IMPROVEMENT_OVER_SOFT_BASELINE'; nxt='D44C_PRIMITIVE_FEATURE_SPACE_REDESIGN'
    if fair_upper['fair_upper_bound_accuracy']<0.97:
        dec='d44b_identifiability_bound_detected'; ver='D44B_PRIMITIVE_IDENTIFIABILITY_BOUND'; nxt='D44C_SUPPORT_EXPANSION_OR_PRIMITIVE_SPACE_REDESIGN'
    else:
        cand=max(['ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID'], key=lambda a:agg['arms'][a]['test_accuracy'])
        cacc=agg['arms'][cand]['test_accuracy']; c_ood=agg['arms'][cand]['OOD_accuracy']
        if cacc>=0.97 and c_ood>=0.97 and (cacc-softb)>=0.02 and tie['candidate_order_sensitivity']<=0.005 and iter_rep['eliminated_correct_candidate_rate']<=0.005 and iter_rep['fair_upper_bound_gap']<=0.03 and agg['arms']['SHUFFLED_CENTER_CONTROL']['test_accuracy']<0.3 and agg['arms']['SHUFFLED_CELL_REFERENCE_CONTROL']['test_accuracy']<0.3 and agg['arms']['NO_CENTER_CONTROL']['test_accuracy']<0.3:
            dec='soft_collision_aware_primitive_discovery_prototype_positive'; ver='D44B_SOFT_PRIMITIVE_DISCOVERY_PROTOTYPE_POSITIVE'; nxt='D44C_PRIMITIVE_DISCOVERY_SCALE_CONFIRM'
        elif max(agg['arms'][a]['test_accuracy'] for a in ['COLLISION_AWARE_SOFT_SCORE','SOFT_PREFILTER_ELIMINATION','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID'])>hardb and max(agg['arms'][a]['test_accuracy'] for a in ['COLLISION_AWARE_SOFT_SCORE','SOFT_PREFILTER_ELIMINATION','ITERATIVE_ELIMINATION_PAIRWISE','SOFT_THEN_HARD_HYBRID'])<=softb:
            dec='soft_primitive_signal_confirmed_but_no_hybrid_gain'; ver='D44B_SOFT_SIGNAL_NO_HYBRID_GAIN'; nxt='D44C_SOFT_PRIMITIVE_SCALE_CONFIRM'
    (out/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2)); (out/'decision.json').write_text(json.dumps({'decision':dec,'verdict':ver,'next':nxt},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2));
    (out/'report.md').write_text('D44B controlled symbolic soft/collision-aware primitive discovery prototype only. No broad claims.\n')

if __name__=='__main__': main()
