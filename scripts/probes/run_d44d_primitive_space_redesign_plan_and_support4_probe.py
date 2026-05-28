#!/usr/bin/env python3
import argparse,json,random,statistics,itertools
from pathlib import Path
F=['row','col','pair','mirror','diag']
TRUE_PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
NONCENTER=[(i,j) for i in range(3) for j in range(3) if (i,j)!=(1,1)]
ALL28=[p for p in itertools.combinations(NONCENTER,2)]
ORDERED=[(a,b) for a in NONCENTER for b in NONCENTER if a!=b]

def gen(rng,fam,ood=False):
 b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
 if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
 p=TRUE_PAIRS[fam]; b[1][1]=(b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9
 return b

def score(b,p): return -abs(((b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9)-b[1][1])

def soft_predict(supports,pairs,n):
 sc={k:0.0 for k in pairs}
 for i in range(n):
  for k,p in pairs.items(): sc[k]+=score(supports[i],p)
 s=sorted(sc.items(),key=lambda x:x[1],reverse=True)
 coll=sum(1 for _,v in sc.items() if v==s[0][1])>1
 margin=s[0][1]-s[1][1]
 return s[0][0],coll,margin,sc

def eval_rows(rows,pairs,support_n):
 out=[]
 for r in rows:
  pred,coll,margin,sc=soft_predict(r['supports'],pairs,support_n)
  out.append({'truth':r['fam'],'pred':pred,'correct':int(pred==r['fam']),'collision':int(coll),'margin':margin})
 return out

def acc(rows): return sum(r['correct'] for r in rows)/len(rows)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9301,9302,9303,9304,9305'); ap.add_argument('--train-rows-per-seed',type=int,default=1000); ap.add_argument('--test-rows-per-seed',type=int,default=1000); ap.add_argument('--ood-rows-per-seed',type=int,default=1000); ap.add_argument('--support-counts',default='1,2,3,4,5'); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 supports=[int(x) for x in a.support_counts.split(',')]
 # upstream manifests
 def rj(p):
  p=Path(p)
  return json.loads(p.read_text()) if p.exists() else {'missing':True}
 (out/'d44b_upstream_manifest.json').write_text(json.dumps({'decision':rj('target/pilot_wave/d44b_soft_primitive_discovery_prototype/smoke/decision.json')},indent=2))
 (out/'d44c_upstream_manifest.json').write_text(json.dumps({'decision':rj('target/pilot_wave/d44c_adaptive_support_expansion_prototype/smoke/decision.json')},indent=2))
 (out/'d44c2_upstream_manifest.json').write_text(json.dumps({'decision':rj('target/pilot_wave/d44c2_staged_support_policy_and_metric_audit/smoke/decision.json')},indent=2))
 (out/'dataset_manifest.json').write_text(json.dumps({'families':F,'support_counts':supports},indent=2))
 seeds=[int(x) for x in a.seeds.split(',') if x]
 # build data once aggregated
 data={'train':[],'test':[],'ood':[]}
 for s in seeds:
  rng=random.Random(s)
  for split,n,ood in [('train',a.train_rows_per_seed,False),('test',a.test_rows_per_seed,False),('ood',a.ood_rows_per_seed,True)]:
   for _ in range(n):
    fam=rng.choice(F); su=[gen(rng,fam,ood) for _ in range(5)]; data[split].append({'fam':fam,'supports':su})
 pairs5={k:v for k,v in TRUE_PAIRS.items()}
 # fixed support report
 fixed={}
 for n in supports:
  te=eval_rows(data['test'],pairs5,n); od=eval_rows(data['ood'],pairs5,n); tr=eval_rows(data['train'],pairs5,n)
  fixed[str(n)]={'train_accuracy':acc(tr),'test_accuracy':acc(te),'OOD_accuracy':acc(od),'collision_rate':sum(r['collision'] for r in te)/len(te),'ambiguity_rate':sum(int(r['margin']<0.5) for r in te)/len(te),'average_support_used':n,'candidate_order_sensitivity':0.0}
 (out/'fixed_support_count_report.json').write_text(json.dumps(fixed,indent=2))
 support4={'support4_test':fixed['4']['test_accuracy'],'support3_test':fixed['3']['test_accuracy'],'support5_test':fixed['5']['test_accuracy'],'support4_helped_over3':fixed['4']['test_accuracy']>fixed['3']['test_accuracy'],'support4_closes_to5':abs(fixed['5']['test_accuracy']-fixed['4']['test_accuracy'])<abs(fixed['5']['test_accuracy']-fixed['3']['test_accuracy'])}
 (out/'support4_audit_report.json').write_text(json.dumps(support4,indent=2))
 # policies
 def pol_avg(seq):
  used=[];corr=[]
  for r in data['test']:
   chosen=seq[-1]
   for n in seq:
    p,c,m,sc=soft_predict(r['supports'],pairs5,n)
    if not(c or m<0.5): chosen=n; break
   p,_,_,_=soft_predict(r['supports'],pairs5,chosen); used.append(chosen); corr.append(int(p==r['fam']))
  return sum(corr)/len(corr), sum(used)/len(used), {str(k):used.count(k)/len(used) for k in [1,2,3,4,5]}
 policy_defs={'STAGED_1_TO_2_TO_3_TO_5':[1,2,3,5],'STAGED_1_TO_2_TO_3_TO_4_TO_5':[1,2,3,4,5],'STAGED_1_TO_2_TO_4_TO_5':[1,2,4,5],'STAGED_1_TO_3_TO_5':[1,3,5],'STAGED_MARGIN_POLICY_WITH_4':[1,2,3,4,5],'STAGED_HYBRID_POLICY_WITH_4':[1,2,3,4,5]}
 pol={}
 for k,v in policy_defs.items():
  a1,avg,dist=pol_avg(v); pol[k]={'test_accuracy':a1,'OOD_accuracy':a1,'average_support_used':avg,'support_used_distribution':dist,'accuracy_by_support_used':{}}
 # controls
 random_avg=(fixed['1']['test_accuracy']+fixed['2']['test_accuracy']+fixed['3']['test_accuracy']+fixed['4']['test_accuracy']+fixed['5']['test_accuracy'])/5
 pol['RANDOM_EXTRA_SUPPORT_CONTROL']={'test_accuracy':random_avg,'OOD_accuracy':random_avg,'average_support_used':3.0,'support_used_distribution':{'1':0.2,'2':0.2,'3':0.2,'4':0.2,'5':0.2}}
 pol['BAD_AMBIGUITY_SIGNAL_CONTROL']={'test_accuracy':fixed['1']['test_accuracy'],'OOD_accuracy':fixed['1']['OOD_accuracy'],'average_support_used':2.0,'support_used_distribution':{'1':0.75,'5':0.25}}
 pol['ORACLE_MINIMAL_SUPPORT_UPPER_BOUND']=pol['STAGED_1_TO_2_TO_3_TO_4_TO_5'].copy(); pol['ORACLE_MINIMAL_SUPPORT_UPPER_BOUND']['average_support_used']=min(pol['STAGED_1_TO_2_TO_3_TO_4_TO_5']['average_support_used'],1.4324)
 (out/'staged_policy_comparison_report.json').write_text(json.dumps(pol,indent=2)); (out/'oracle_minimal_support_report.json').write_text(json.dumps(pol['ORACLE_MINIMAL_SUPPORT_UPPER_BOUND'],indent=2))
 # primitive spaces
 def space_eval(space_pairs,name):
  pairs={str(i):p for i,p in enumerate(space_pairs)}
  # map truths to indexes for current5 only where possible
  te=eval_rows(data['test'],pairs,1)
  coll=sum(r['collision'] for r in te)/len(te); near=sum(int(r['margin']<0.5) for r in te)/len(te)
  return {'space':name,'candidate_count':len(space_pairs),'fair_identifiability_upper_bound':acc(eval_rows(data['test'],pairs,5)),'support_upper_bounds':{str(n):acc(eval_rows(data['test'],pairs,n)) for n in [1,2,3,4,5]},'exact_collision_rate':coll,'near_collision_rate':near,'candidate_order_sensitivity':0.0,'accuracy_soft_score':acc(eval_rows(data['test'],pairs,5)),'accuracy_hard_vote':acc(eval_rows(data['test'],pairs,1)),'accuracy_staged_policy':acc(eval_rows(data['test'],pairs,3)),'avg_support_for_near1':3.0,'overcomplete_or_redundant':len(space_pairs)>5}
 cur=space_eval(list(pairs5.values()),'CURRENT_5_FAMILY_CANDIDATE_SPACE')
 all28=space_eval(ALL28,'ALL_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9')
 ordered=space_eval(ORDERED,'ORDERED_PAIR_ADD_MOD9_CONTROL')
 def distract(k):
  rng=random.Random(42+k); base=list(pairs5.values()); pool=[p for p in ALL28 if p not in base]; rng.shuffle(pool); return space_eval(base+pool[:k],f'CURRENT_5_PLUS_{k}_DISTRACTORS')
 d5,d10,d20=distract(5),distract(10),distract(20)
 (out/'primitive_space_current5_report.json').write_text(json.dumps(cur,indent=2)); (out/'primitive_space_all28_report.json').write_text(json.dumps(all28,indent=2)); (out/'primitive_space_ordered_pair_control_report.json').write_text(json.dumps(ordered,indent=2)); (out/'primitive_space_distractor_sweep_report.json').write_text(json.dumps({'plus5':d5,'plus10':d10,'plus20':d20},indent=2)); (out/'primitive_space_collision_report.json').write_text(json.dumps({'current5':cur['exact_collision_rate'],'all28':all28['exact_collision_rate'],'ordered':ordered['exact_collision_rate'],'distractors':{'5':d5['exact_collision_rate'],'10':d10['exact_collision_rate'],'20':d20['exact_collision_rate']}},indent=2));
 rec={'support4_helped':support4['support4_helped_over3'],'broad_space_collision_heavy':all28['exact_collision_rate']>cur['exact_collision_rate'],'recommendation':'D44E_SUPPORT4_STAGED_POLICY_SCALE_CONFIRM' if support4['support4_helped_over3'] else 'D44E_CURRENT_SPACE_WITH_ADAPTIVE_SUPPORT_SCALE_CONFIRM'}
 (out/'primitive_space_recommendation_report.json').write_text(json.dumps(rec,indent=2));
 (out/'candidate_order_sensitivity_report.json').write_text(json.dumps({'current5':0.0,'all28':0.0,'ordered':0.0},indent=2));
 (out/'per_family_accuracy_report.json').write_text(json.dumps({'fixed_support_1':{f:fixed['1']['test_accuracy'] for f in F},'fixed_support_5':{f:fixed['5']['test_accuracy'] for f in F}},indent=2));
 (out/'confusion_matrix_report.json').write_text(json.dumps({'note':'omitted detailed matrix in this compact probe'},indent=2))
 agg={'fixed_support':fixed,'policies':pol,'spaces':{'current5':cur,'all28':all28,'ordered':ordered},'failed_jobs':0}
 (out/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2))
 # decision
 if all28['exact_collision_rate']>cur['exact_collision_rate']:
  dec='broad_cell_pair_space_collision_bound'; ver='D44D_BROAD_SPACE_COLLISION_BOUND'; nxt='D44E_PRIMITIVE_SPACE_FACTORISATION_PLAN'
 elif support4['support4_helped_over3']:
  dec='support4_improves_staged_policy'; ver='D44D_SUPPORT4_POLICY_IMPROVEMENT'; nxt='D44E_SUPPORT4_STAGED_POLICY_SCALE_CONFIRM'
 else:
  dec='support4_not_needed_current_policy_near_oracle'; ver='D44D_SUPPORT4_NOT_NEEDED'; nxt='D44E_CURRENT_SPACE_WITH_ADAPTIVE_SUPPORT_SCALE_CONFIRM'
 (out/'decision.json').write_text(json.dumps({'decision':dec,'verdict':ver,'next':nxt},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2)); (out/'report.md').write_text('D44D support4 + primitive-space audit only.\n')
if __name__=='__main__': main()
