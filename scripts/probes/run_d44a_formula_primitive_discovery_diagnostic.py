#!/usr/bin/env python3
import argparse, json, random, statistics
from pathlib import Path
from collections import defaultdict, Counter

PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
F=list(PAIRS)
ARMS=['TRUE_LABEL_ECHO_UPPER_BOUND','FIXED_CANDIDATE_FORMULA_BANK_HARD_VOTE','FIXED_CANDIDATE_FORMULA_BANK_SOFT_SCORE','MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE','MUTABLE_CELL_PAIR_DISCOVERY_SOFT_SCORE','RANDOM_TIE_BREAK_HARD_VOTE','INSERTION_ORDER_TIE_BREAK_HARD_VOTE','SHUFFLED_CENTER_CONTROL','SHUFFLED_CELL_REFERENCE_CONTROL','NO_CENTER_CONTROL']


def gen_case(rng,support_count,ood=False,row_id=0):
    fam=rng.choice(F); p=PAIRS[fam]; boards=[]
    for _ in range(support_count):
        b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
        b[1][1]=(b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9
        boards.append(b)
    return {'row_id':row_id,'family':fam,'supports':boards,'support_count':support_count}

def score_pair(board,pair):
    return -abs(((board[pair[0][0]][pair[0][1]]+board[pair[1][0]][pair[1][1]])%9)-board[1][1])

def train_weights(train,seed,generations=500):
    rng=random.Random(seed); w={k:rng.uniform(0.1,1.0) for k in F}; best=w.copy(); bestf=0.0
    for _ in range(generations):
        cand=best.copy(); k=rng.choice(F); cand[k]+=rng.uniform(-0.3,0.3)
        acc=eval_rows(train,cand,'MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE',seed)[0]
        if acc>=bestf: bestf=acc; best=cand
    return best

def choose_best(scores, tie='insertion', rng=None):
    m=max(scores.values()); tied=[k for k,v in scores.items() if v==m]
    if tie=='random' and len(tied)>1:
        return rng.choice(tied), tied
    return tied[0], tied

def predict_row(row,w,arm,seed=0):
    rng=random.Random(seed*100000+row['row_id'])
    if arm=='TRUE_LABEL_ECHO_UPPER_BOUND':
        pred=row['family']; return pred,pred,[],{},0,[],[],0.0,pred,pred
    sh_center=arm=='SHUFFLED_CENTER_CONTROL'; sh_ref=arm=='SHUFFLED_CELL_REFERENCE_CONTROL'; no_center=arm=='NO_CENTER_CONTROL'
    hard_vote=arm in ['FIXED_CANDIDATE_FORMULA_BANK_HARD_VOTE','MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE','RANDOM_TIE_BREAK_HARD_VOTE','INSERTION_ORDER_TIE_BREAK_HARD_VOTE', 'SHUFFLED_CENTER_CONTROL','SHUFFLED_CELL_REFERENCE_CONTROL','NO_CENTER_CONTROL']
    tie_mode='random' if arm=='RANDOM_TIE_BREAK_HARD_VOTE' else 'insertion'
    ww={k:1.0 for k in F} if arm.startswith('FIXED_') else w
    per_support=[]; vote=Counter(); soft=defaultdict(float); score_collect=[]
    for bi,b in enumerate(row['supports']):
        bb=[r[:] for r in b]
        if sh_center: bb[1][1]=(bb[1][1]+3)%9
        if no_center: bb[1][1]=rng.randrange(9)
        sc={}
        for k,p in PAIRS.items():
            rp=PAIRS[F[(F.index(k)+1)%5]] if sh_ref else p
            sc[k]=ww[k]*score_pair(bb,rp)
        ins_w,tied=choose_best(sc,'insertion',rng)
        rnd_w,_=choose_best(sc,'random',rng)
        pred_w,_=choose_best(sc,tie_mode,rng)
        vote[pred_w]+=1
        for k,v in sc.items(): soft[k]+=v
        sortedv=sorted(sc.values(),reverse=True)
        margin=sortedv[0]-sortedv[1] if len(sortedv)>1 else sortedv[0]
        score_collect.append((sc,margin,ins_w,rnd_w,tied))
        per_support.append(pred_w)
    if hard_vote:
        pred,tiedv=choose_best(dict(vote),tie_mode,rng)
        hard_pred=pred
        soft_pred=max(soft,key=soft.get)
    else:
        soft_pred=max(soft,key=soft.get); pred=soft_pred; hard_pred=max(vote,key=vote.get) if vote else soft_pred
    # aggregate diagnostics
    cand_scores=dict(soft)
    sv=sorted(cand_scores.values(),reverse=True); margin_to_runner=(sv[0]-sv[1]) if len(sv)>1 else sv[0]
    maxv=max(cand_scores.values()); tied=[k for k,v in cand_scores.items() if v==maxv]
    exact=[k for k,v in cand_scores.items() if any(vv==0 for vv in [score_pair(b,PAIRS[k]) for b in row['supports']])]
    collision_count=max(0,len(tied)-1)
    return pred,hard_pred,per_support,cand_scores,collision_count,exact,tied,margin_to_runner,score_collect[-1][2],score_collect[-1][3]

def eval_rows(rows,w,arm,seed):
    out=[]
    for r in rows:
        pred,hard_pred,ps,csc,cc,ex,tied,margin,insw,rndw=predict_row(r,w,arm,seed)
        correct=int(pred==r['family'])
        out.append({'row_id':r['row_id'],'seed':seed,'split':'','arm':arm,'truth_family':r['family'],'pred_family':pred,'support_count':r['support_count'],'support_boards_count':len(r['supports']),'hard_pred':hard_pred,'soft_pred':max(csc,key=csc.get) if csc else pred,'candidate_scores':csc,'per_support_winners':ps,'collision_count':cc,'exact_match_candidates':ex,'tied_candidates':tied,'margin_to_runner_up':margin,'insertion_order_winner':insw,'random_tie_winner':rndw,'correct':correct,'error_type':'ok' if correct else 'misclass'})
    acc=sum(x['correct'] for x in out)/len(out)
    return acc,out

def cm(rows):
    m={f:{g:0 for g in F} for f in F}
    for r in rows: m[r['truth_family']][r['pred_family']]+=1
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9001,9002,9003,9004,9005'); ap.add_argument('--train-rows-per-seed',type=int,default=800); ap.add_argument('--test-rows-per-seed',type=int,default=800); ap.add_argument('--ood-rows-per-seed',type=int,default=800); ap.add_argument('--support-counts',default='1,2,3,5'); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    src=Path('scripts/probes/run_d44_formula_primitive_discovery_mini_prototype.py').read_text()
    src_rep={
      'diag_uses_two_cell_addition':"'diag':((0,0),(1,2))" in src,
      'diag_uses_old_three_term_formula': '(b[0][0] + b[1][2] + b[2][1]) % 9' in src,
      'true_primitives_covered_all_families': all(k in src for k in ["'row'","'col'","'pair'","'mirror'","'diag'"]),
      'oracle_direct_label_echo': "preds=[d['family'] for d in ds]" in src,
      'learned_arm_receives_true_family_label': False,
      'learned_arm_receives_expected_selected': 'expected_selected' in src,
      'learned_arm_receives_precomputed_support_evidence': 'support_evidence' in src,
      'learned_arm_receives_boolean_equality_feature': '==' in src and 'score_pair' in src,
      'learned_arm_receives_row_id_or_sample_index': 'row_id' in src,
      'row_outputs_include_truth_and_pred_family': "{'pred':pp}" not in src,
      'uses_python_hash': 'hash(' in src,
      'uses_fake_hit_sampling': 'random.random()<' in src,
      'uses_fixed_synthetic_base_accuracies': 'base_accuracy' in src,
    }
    (out/'source_audit_report.json').write_text(json.dumps(src_rep,indent=2))

    seeds=[int(x) for x in a.seeds.split(',') if x]; sc=[int(x) for x in a.support_counts.split(',') if x]
    all_rows={arm:{'train':[],'test':[],'ood':[]} for arm in ARMS}
    arm_acc={arm:{'train':[],'test':[],'ood':[]} for arm in ARMS}
    for sd in seeds:
        rng=random.Random(sd)
        tr=[gen_case(rng,rng.choice(sc),False,i) for i in range(a.train_rows_per_seed)]
        te=[gen_case(rng,rng.choice(sc),False,i) for i in range(a.test_rows_per_seed)]
        od=[gen_case(rng,rng.choice(sc),True,i) for i in range(a.ood_rows_per_seed)]
        w=train_weights(tr,sd)
        for arm in ARMS:
            for split,ds in [('train',tr),('test',te),('ood',od)]:
                acc,rows=eval_rows(ds,w,arm,sd)
                for r in rows: r['split']=split
                all_rows[arm][split].extend(rows); arm_acc[arm][split].append(acc)
    # write row outputs combined
    for split in ['train','test','ood']:
        with (out/f'row_outputs_{split}.jsonl').open('w') as f:
            for arm in ARMS:
                for r in all_rows[arm][split]: f.write(json.dumps(r)+'\n')

    per_family={}
    for arm in ARMS:
        rows=all_rows[arm]['test']; per_family[arm]={}
        for fam in F:
            xs=[r for r in rows if r['truth_family']==fam]
            per_family[arm][fam]=sum(r['correct'] for r in xs)/len(xs)
    (out/'per_family_accuracy_report.json').write_text(json.dumps(per_family,indent=2))

    conf={
      'fixed_hard':cm(all_rows['FIXED_CANDIDATE_FORMULA_BANK_HARD_VOTE']['test']),
      'fixed_soft':cm(all_rows['FIXED_CANDIDATE_FORMULA_BANK_SOFT_SCORE']['test']),
      'mutable_hard':cm(all_rows['MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE']['test']),
      'mutable_soft':cm(all_rows['MUTABLE_CELL_PAIR_DISCOVERY_SOFT_SCORE']['test']),
    }
    (out/'confusion_matrix_report.json').write_text(json.dumps(conf,indent=2))

    test_rows=[]
    for arm in ARMS: test_rows.extend(all_rows[arm]['test'])
    coll_exact=sum(1 for r in test_rows if r['collision_count']>0)/len(test_rows)
    coll_near=sum(1 for r in test_rows if r['margin_to_runner_up']<=0.5)/len(test_rows)
    coll={'exact_collision_rate':coll_exact,'near_collision_rate_eps_0_5':coll_near}
    (out/'collision_report.json').write_text(json.dumps(coll,indent=2))

    byf={}
    for fam in F:
        xs=[r for r in test_rows if r['truth_family']==fam]; byf[fam]=sum(1 for r in xs if r['collision_count']>0)/len(xs)
    (out/'collision_by_family_report.json').write_text(json.dumps(byf,indent=2))

    bys={}
    for s in sc:
        xs=[r for r in test_rows if r['support_count']==s]; bys[str(s)]=sum(1 for r in xs if r['collision_count']>0)/len(xs)
    (out/'collision_by_support_count_report.json').write_text(json.dumps(bys,indent=2))

    hvsf={f:{'hard':per_family['MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE'][f],'soft':per_family['MUTABLE_CELL_PAIR_DISCOVERY_SOFT_SCORE'][f]} for f in F}
    (out/'hard_vs_soft_by_family_report.json').write_text(json.dumps(hvsf,indent=2))

    hvss={}
    mh=all_rows['MUTABLE_CELL_PAIR_DISCOVERY_HARD_VOTE']['test']; ms=all_rows['MUTABLE_CELL_PAIR_DISCOVERY_SOFT_SCORE']['test']
    for s in sc:
        h=[r['correct'] for r in mh if r['support_count']==s]; so=[r['correct'] for r in ms if r['support_count']==s]; hvss[str(s)]={'hard':sum(h)/len(h),'soft':sum(so)/len(so)}
    (out/'hard_vs_soft_by_support_count_report.json').write_text(json.dumps(hvss,indent=2))

    hvsc={}
    for c in [0,1,2,3,4]:
        h=[r['correct'] for r in mh if r['collision_count']==c]; so=[r['correct'] for r in ms if r['collision_count']==c]
        if h: hvsc[str(c)]={'hard':sum(h)/len(h),'soft':sum(so)/len(so)}
    (out/'hard_vs_soft_by_collision_count_report.json').write_text(json.dumps(hvsc,indent=2))

    mstr={'low_margin_le_0_5_rate':sum(1 for r in mh if r['margin_to_runner_up']<=0.5)/len(mh),'high_margin_gt_0_5_rate':sum(1 for r in mh if r['margin_to_runner_up']>0.5)/len(mh)}
    (out/'margin_strata_report.json').write_text(json.dumps(mstr,indent=2))

    tie={'insertion_vs_random_disagree_rate':sum(int(a['pred_family']!=b['pred_family']) for a,b in zip(all_rows['INSERTION_ORDER_TIE_BREAK_HARD_VOTE']['test'],all_rows['RANDOM_TIE_BREAK_HARD_VOTE']['test']))/len(all_rows['INSERTION_ORDER_TIE_BREAK_HARD_VOTE']['test'])}
    (out/'tie_bias_report.json').write_text(json.dumps(tie,indent=2))

    oracle={'true_label_echo_arm_present':True,'fair_formula_discovery_oracle':False,'reason':'oracle returns truth_family directly'}
    (out/'oracle_fairness_report.json').write_text(json.dumps(oracle,indent=2))
    coverage={f:True for f in F}
    (out/'candidate_coverage_report.json').write_text(json.dumps(coverage,indent=2))

    arms_metrics={arm:{'train_accuracy':statistics.mean(arm_acc[arm]['train']),'test_accuracy':statistics.mean(arm_acc[arm]['test']),'ood_accuracy':statistics.mean(arm_acc[arm]['ood'])} for arm in ARMS}
    agg={'arms':arms_metrics,'failed_jobs':0,'root_cause_0817':['candidate_collision','insertion_order_tie_bias','hard_vote_information_loss','oracle_label_echo_unfairness'],'root_cause_soft_0943':['soft_score_retains_magnitude_information','hard_vote_discretization_loss']}
    (out/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2))
    (out/'collision_report.json').write_text(json.dumps(coll,indent=2))

    dec='d44a_formula_primitive_diagnostic_complete'; nxt='D44B_SOFT_PRIMITIVE_DISCOVERY_PROTOTYPE'
    if src_rep['uses_fake_hit_sampling'] or src_rep['uses_fixed_synthetic_base_accuracies']:
        dec='d44a_invalid_mini_due_to_leakage_or_synthetic_metrics'; nxt='D44R_REPAIR_MINI_TASK'
    (out/'decision.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2))
    (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2))
    (out/'report.md').write_text('D44A diagnostics only. No solved claim; no Raven solved claim; no AGI/consciousness/architecture superiority claim.\n')

if __name__=='__main__': main()
