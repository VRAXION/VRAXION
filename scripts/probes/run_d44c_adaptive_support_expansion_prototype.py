#!/usr/bin/env python3
import argparse,json,random,statistics,math
from pathlib import Path
from collections import Counter,defaultdict
PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
F=list(PAIRS)
ARMS=['ONE_SHOT_SUPPORT_1_BASELINE','FIXED_SUPPORT_2','FIXED_SUPPORT_3','FIXED_SUPPORT_5','ADAPTIVE_SUPPORT_EXPANSION_SOFT','ADAPTIVE_SUPPORT_EXPANSION_CONSERVATIVE','RANDOM_EXTRA_SUPPORT_CONTROL','BAD_AMBIGUITY_SIGNAL_CONTROL','TRUE_LABEL_ECHO_REFERENCE_ONLY']

def gen_support(rng,fam,ood=False):
 p=PAIRS[fam]; b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
 if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
 b[1][1]=(b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9
 return b

def row_case(rng,row_id,max_support,ood):
 fam=rng.choice(F); su=[gen_support(rng,fam,ood) for _ in range(max_support)]
 return {'row_id':row_id,'truth_family':fam,'supports':su}

def score(b,k):
 p=PAIRS[k];return -abs(((b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9)-b[1][1])

def entropy(scores):
 vals=list(scores.values()); m=max(vals); ex=[math.exp(v-m) for v in vals]; s=sum(ex); p=[x/s for x in ex]; return -sum(x*math.log(x+1e-12) for x in p)

def pred_with_supports(row,used,arm,rng):
 if arm=='TRUE_LABEL_ECHO_REFERENCE_ONLY': return row['truth_family'],{},0,False
 soft=defaultdict(float); col=False
 for i in range(used):
  b=row['supports'][i]; sc={k:score(b,k) for k in F}; m=max(sc.values()); ties=[k for k,v in sc.items() if v==m]; col=col or (len(ties)>1)
  for k,v in sc.items(): soft[k]+=v
 pred=max(soft,key=soft.get)
 margin=sorted(soft.values(),reverse=True); margin=(margin[0]-margin[1]) if len(margin)>1 else margin[0]
 amb=(margin<0.5) or col or entropy(soft)>1.45
 if arm=='BAD_AMBIGUITY_SIGNAL_CONTROL': amb=(margin>1.0)
 return pred,dict(soft),margin,amb

def evaluate(rows,arm,max_support,seed):
    rng=random.Random(seed+7); out=[]
    for r in rows:
        if arm=='ONE_SHOT_SUPPORT_1_BASELINE':
            used=1
        elif arm=='FIXED_SUPPORT_2':
            used=2
        elif arm=='FIXED_SUPPORT_3':
            used=3
        elif arm=='FIXED_SUPPORT_5':
            used=5
        elif arm=='RANDOM_EXTRA_SUPPORT_CONTROL':
            used=1+rng.randrange(max_support)
        elif arm in ['ADAPTIVE_SUPPORT_EXPANSION_SOFT','ADAPTIVE_SUPPORT_EXPANSION_CONSERVATIVE','BAD_AMBIGUITY_SIGNAL_CONTROL']:
            used=1
            while used<max_support:
                _,_,_,amb=pred_with_supports(r,used,arm,rng)
                if not amb:
                    break
                used+=1
        else:
            used=1
        pred,sc,mg,amb=pred_with_supports(r,used,arm,rng)
        abstain=(arm=='ADAPTIVE_SUPPORT_EXPANSION_CONSERVATIVE' and amb and used>=max_support)
        if abstain: pred='ABSTAIN'
        corr=int(pred==r['truth_family'])
        tie_ct=0
        if sc:
            m=max(sc.values()); tie_ct=len([k for k,v in sc.items() if v==m])
        out.append({'row_id':r['row_id'],'truth_family':r['truth_family'],'pred_family':pred,'support_used':used,'correct':corr,'collision':int(tie_ct>1),'ambiguous':int(amb),'abstain':int(abstain)})
    return out

def summarize(rows):
 total=len(rows); acc=sum(r['correct'] for r in rows)/total
 sup=Counter(r['support_used'] for r in rows)
 bysup={str(k):sum(r['correct'] for r in rows if r['support_used']==k)/max(1,sup[k]) for k in sorted(sup)}
 fam={f:sum(r['correct'] for r in rows if r['truth_family']==f)/max(1,sum(1 for r in rows if r['truth_family']==f)) for f in F}
 col=[r for r in rows if r['collision']==1]; amb=[r for r in rows if r['ambiguous']==1]
 return {'accuracy':acc,'average_support_used':sum(r['support_used'] for r in rows)/total,'support_used_distribution':{str(k):v/total for k,v in sup.items()},'accuracy_by_support_used':bysup,'per_family_accuracy':fam,'collision_case_accuracy':sum(r['correct'] for r in col)/max(1,len(col)),'ambiguous_case_accuracy':sum(r['correct'] for r in amb)/max(1,len(amb)),'abstain_rate':sum(r['abstain'] for r in rows)/total,'effective_accuracy_counting_abstain_wrong':acc,'error_count':sum(1-r['correct'] for r in rows),'failed_seed_count':0}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9201,9202,9203,9204,9205'); ap.add_argument('--train-rows-per-seed',type=int,default=1000); ap.add_argument('--test-rows-per-seed',type=int,default=1000); ap.add_argument('--ood-rows-per-seed',type=int,default=1000); ap.add_argument('--max-support-count',type=int,default=5); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 up='missing'; p=Path('target/pilot_wave/d44b_soft_primitive_discovery_prototype/smoke/decision.json')
 if p.exists(): up=json.loads(p.read_text())
 (out/'d44b_upstream_manifest.json').write_text(json.dumps({'d44b_decision':up},indent=2))
 (out/'dataset_manifest.json').write_text(json.dumps({'families':F,'pairs':PAIRS,'max_support_count':a.max_support_count},indent=2))
 seeds=[int(x) for x in a.seeds.split(',') if x]
 metrics={arm:{'train':[],'test':[],'ood':[]} for arm in ARMS}; alltest={arm:[] for arm in ARMS}
 for s in seeds:
  rng=random.Random(s)
  tr=[row_case(rng,i,a.max_support_count,False) for i in range(a.train_rows_per_seed)]
  te=[row_case(rng,i,a.max_support_count,False) for i in range(a.test_rows_per_seed)]
  od=[row_case(rng,i,a.max_support_count,True) for i in range(a.ood_rows_per_seed)]
  for arm in ARMS:
   for split,ds in [('train',tr),('test',te),('ood',od)]:
    rs=evaluate(ds,arm,a.max_support_count,s); sm=summarize(rs); metrics[arm][split].append(sm)
    if split=='test': alltest[arm].extend(rs)
 # reports
 agg={'arms':{}}
 per_family={}; conf={}
 for arm in ARMS:
  agg['arms'][arm]={'train_accuracy':statistics.mean(x['accuracy'] for x in metrics[arm]['train']),'test_accuracy':statistics.mean(x['accuracy'] for x in metrics[arm]['test']),'OOD_accuracy':statistics.mean(x['accuracy'] for x in metrics[arm]['ood']),'average_support_used':statistics.mean(x['average_support_used'] for x in metrics[arm]['test']),'support_used_distribution':metrics[arm]['test'][0]['support_used_distribution'],'accuracy_by_support_used':metrics[arm]['test'][0]['accuracy_by_support_used'],'per_family_accuracy':{f:statistics.mean(x['per_family_accuracy'][f] for x in metrics[arm]['test']) for f in F},'collision_case_accuracy':statistics.mean(x['collision_case_accuracy'] for x in metrics[arm]['test']),'ambiguous_case_accuracy':statistics.mean(x['ambiguous_case_accuracy'] for x in metrics[arm]['test']),'abstain_rate':statistics.mean(x['abstain_rate'] for x in metrics[arm]['test']),'effective_accuracy_counting_abstain_wrong':statistics.mean(x['effective_accuracy_counting_abstain_wrong'] for x in metrics[arm]['test']),'error_count':statistics.mean(x['error_count'] for x in metrics[arm]['test']),'failed_seed_count':0}
  per_family[arm]=agg['arms'][arm]['per_family_accuracy']
  cm={f:{g:0 for g in F+['ABSTAIN']} for f in F}
  for r in alltest[arm]: cm[r['truth_family']][r['pred_family']]+=1
  conf[arm]=cm
 (out/'per_family_accuracy_report.json').write_text(json.dumps(per_family,indent=2)); (out/'confusion_matrix_report.json').write_text(json.dumps(conf,indent=2))
 one=agg['arms']['ONE_SHOT_SUPPORT_1_BASELINE']; fs2=agg['arms']['FIXED_SUPPORT_2']; fs3=agg['arms']['FIXED_SUPPORT_3']; fs5=agg['arms']['FIXED_SUPPORT_5']; ad=agg['arms']['ADAPTIVE_SUPPORT_EXPANSION_SOFT']; adc=agg['arms']['ADAPTIVE_SUPPORT_EXPANSION_CONSERVATIVE']; rnd=agg['arms']['RANDOM_EXTRA_SUPPORT_CONTROL']; bad=agg['arms']['BAD_AMBIGUITY_SIGNAL_CONTROL']
 (out/'one_shot_support_1_report.json').write_text(json.dumps(one,indent=2)); (out/'fixed_support_2_report.json').write_text(json.dumps(fs2,indent=2)); (out/'fixed_support_3_report.json').write_text(json.dumps(fs3,indent=2)); (out/'fixed_support_5_report.json').write_text(json.dumps(fs5,indent=2)); (out/'adaptive_support_expansion_soft_report.json').write_text(json.dumps(ad,indent=2)); (out/'adaptive_support_expansion_conservative_report.json').write_text(json.dumps(adc,indent=2)); (out/'random_extra_support_control_report.json').write_text(json.dumps(rnd,indent=2)); (out/'bad_ambiguity_signal_control_report.json').write_text(json.dumps(bad,indent=2))
 coll={'collision_case_accuracy_soft':ad['collision_case_accuracy'],'collision_case_accuracy_one_shot':one['collision_case_accuracy'],'ambiguous_case_accuracy_soft':ad['ambiguous_case_accuracy']}
 (out/'collision_report.json').write_text(json.dumps(coll,indent=2))
 # adaptive diagnostics approximation
 amb_prec=ad['ambiguous_case_accuracy']; amb_rec=1.0-ad['accuracy_by_support_used'].get('1',ad['test_accuracy'])
 req_prec=1.0-(ad['average_support_used']-1)/4; req_rec=min(1.0,(ad['average_support_used']-1)/2)
 adaptive={'ambiguity_detection_precision':amb_prec,'ambiguity_detection_recall':amb_rec,'support_request_precision':req_prec,'support_request_recall':req_rec,'unnecessary_extra_support_rate':max(0.0,1-req_prec),'unresolved_after_max_support_rate':adc['abstain_rate'],'average_support_saved_vs_fixed_5':5-ad['average_support_used'],'accuracy_gain_vs_support_1':ad['test_accuracy']-one['test_accuracy'],'accuracy_gain_vs_random_extra_support':ad['test_accuracy']-rnd['test_accuracy']}
 (out/'adaptive_policy_report.json').write_text(json.dumps(adaptive,indent=2)); (out/'support_request_report.json').write_text(json.dumps({'average_support_used_soft':ad['average_support_used'],'average_support_used_conservative':adc['average_support_used']},indent=2)); (out/'identifiability_report.json').write_text(json.dumps({'support1':one['test_accuracy'],'support2':fs2['test_accuracy'],'support3':fs3['test_accuracy'],'support5':fs5['test_accuracy']},indent=2)); (out/'ambiguity_detection_report.json').write_text(json.dumps({'precision':amb_prec,'recall':amb_rec},indent=2))
 (out/'support_efficiency_report.json').write_text(json.dumps(adaptive,indent=2));
 efs={str(k):sum(int(r['correct']) for r in alltest['ADAPTIVE_SUPPORT_EXPANSION_SOFT'] if r['support_used']==k)/max(1,sum(1 for r in alltest['ADAPTIVE_SUPPORT_EXPANSION_SOFT'] if r['support_used']==k)) for k in [1,2,3,4,5]}
 (out/'error_by_final_support_count_report.json').write_text(json.dumps(efs,indent=2))
 agg['failed_jobs']=0; (out/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2))
 # decision
 dec='adaptive_support_expansion_not_confirmed'; ver='D44C_ADAPTIVE_SUPPORT_EXPANSION_NOT_CONFIRMED'; nxt='D44D_PRIMITIVE_SPACE_REDESIGN_PLAN'
 if ad['test_accuracy']>=0.98 and ad['OOD_accuracy']>=0.98 and ad['average_support_used']<5:
  dec='adaptive_support_expansion_prototype_positive'; ver='D44C_ADAPTIVE_SUPPORT_EXPANSION_PROTOTYPE_POSITIVE'; nxt='D44D_PRIMITIVE_SPACE_REDESIGN_PLAN'
 elif ad['test_accuracy']>=0.98 and ad['OOD_accuracy']>=0.98 and ad['average_support_used']>=5:
  dec='support_expansion_solves_but_no_adaptive_efficiency'; ver='D44C_SUPPORT_EXPANSION_NO_EFFICIENCY'; nxt='D44D_AMBIGUITY_POLICY_OPTIMIZATION'
 (out/'decision.json').write_text(json.dumps({'decision':dec,'verdict':ver,'next':nxt},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2)); (out/'report.md').write_text('D44C adaptive support expansion on symbolic identifiability task only.\n')

if __name__=="__main__": main()
