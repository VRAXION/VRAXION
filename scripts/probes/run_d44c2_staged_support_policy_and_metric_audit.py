#!/usr/bin/env python3
import argparse,json,random,statistics,math
from pathlib import Path
from collections import defaultdict
PAIRS={'row':((1,0),(1,2)),'col':((0,1),(2,1)),'pair':((0,0),(2,2)),'mirror':((2,0),(0,2)),'diag':((0,0),(1,2))}
F=list(PAIRS)
ARMS=['ONE_SHOT_SUPPORT_1_BASELINE','FIXED_SUPPORT_2','FIXED_SUPPORT_3','FIXED_SUPPORT_5','OLD_ADAPTIVE_1_TO_5_REPLAY','STAGED_SUPPORT_1_TO_2_TO_3_TO_5','STAGED_SUPPORT_MARGIN_POLICY','STAGED_SUPPORT_ENTROPY_POLICY','STAGED_SUPPORT_HYBRID_POLICY','RANDOM_EXTRA_SUPPORT_CONTROL','BAD_AMBIGUITY_SIGNAL_CONTROL','ORACLE_MINIMAL_SUPPORT_UPPER_BOUND']

def gen(rng,fam,ood=False):
 b=[[rng.randrange(9) for _ in range(3)] for _ in range(3)]
 if ood: b=[[((x*2)+1)%9 for x in r] for r in b]
 p=PAIRS[fam]; b[1][1]=(b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9
 return b

def score(b,k):
 p=PAIRS[k]; return -abs(((b[p[0][0]][p[0][1]]+b[p[1][0]][p[1][1]])%9)-b[1][1])

def entropy(sc):
 vals=list(sc.values()); m=max(vals); ex=[math.exp(v-m) for v in vals]; s=sum(ex); p=[x/s for x in ex]; return -sum(x*math.log(x+1e-12) for x in p)

def pred_supports(supports,n):
 soft=defaultdict(float)
 for i in range(n):
  sc={k:score(supports[i],k) for k in F}
  for k,v in sc.items(): soft[k]+=v
 srt=sorted(soft.items(),key=lambda x:x[1],reverse=True)
 margin=srt[0][1]-srt[1][1]; coll=int(srt[0][1]==srt[1][1]); ent=entropy(soft)
 return srt[0][0],margin,coll,ent

def oracle_min_needed(supports,truth):
 for n in [1,2,3,5]:
  p,m,c,e=pred_supports(supports,n)
  if p==truth and (m>=0.5) and c==0: return n
 return 5

def run_row(row,arm,rng):
 t=row['truth']; su=row['supports']; used=1; req=[]
 if arm=='ONE_SHOT_SUPPORT_1_BASELINE': used=1
 elif arm=='FIXED_SUPPORT_2': used=2
 elif arm=='FIXED_SUPPORT_3': used=3
 elif arm=='FIXED_SUPPORT_5': used=5
 elif arm=='OLD_ADAPTIVE_1_TO_5_REPLAY':
  p,m,c,e=pred_supports(su,1); need=(m<0.5 or c or e>1.45); req=[1,int(need)]
  used=5 if need else 1
 elif arm=='STAGED_SUPPORT_1_TO_2_TO_3_TO_5':
  used=1
  for nxt in [2,3,5]:
   p,m,c,e=pred_supports(su,used); need=(m<0.5 or c or e>1.45); req.append((used,int(need)))
   if not need: break
   used=nxt
 elif arm=='STAGED_SUPPORT_MARGIN_POLICY':
  used=1
  for nxt in [2,3,5]:
   p,m,c,e=pred_supports(su,used); need=(m<0.8); req.append((used,int(need)))
   if not need: break
   used=nxt
 elif arm=='STAGED_SUPPORT_ENTROPY_POLICY':
  used=1
  for nxt in [2,3,5]:
   p,m,c,e=pred_supports(su,used); need=(e>1.35); req.append((used,int(need)))
   if not need: break
   used=nxt
 elif arm=='STAGED_SUPPORT_HYBRID_POLICY':
  used=1
  for nxt in [2,3,5]:
   p,m,c,e=pred_supports(su,used); need=(m<0.6 or c or e>1.3); req.append((used,int(need)))
   if not need: break
   used=nxt
 elif arm=='RANDOM_EXTRA_SUPPORT_CONTROL': used=rng.choice([1,2,3,5])
 elif arm=='BAD_AMBIGUITY_SIGNAL_CONTROL':
  p,m,c,e=pred_supports(su,1); used=5 if m>1.0 else 1
 elif arm=='ORACLE_MINIMAL_SUPPORT_UPPER_BOUND': used=oracle_min_needed(su,t)
 p,m,c,e=pred_supports(su,used)
 return {'truth_family':t,'pred_family':p,'support_used':used,'correct':int(p==t),'collision':c,'ambiguous':int(m<0.5 or c or e>1.45),'request_history':req}

def summarize(rows):
 n=len(rows); acc=sum(r['correct'] for r in rows)/n
 dist={str(k):sum(1 for r in rows if r['support_used']==k)/n for k in [1,2,3,5]}
 by={str(k):(sum(r['correct'] for r in rows if r['support_used']==k)/max(1,sum(1 for r in rows if r['support_used']==k))) for k in [1,2,3,5]}
 pf={f:sum(r['correct'] for r in rows if r['truth_family']==f)/max(1,sum(1 for r in rows if r['truth_family']==f)) for f in F}
 col=[r for r in rows if r['collision']==1]; amb=[r for r in rows if r['ambiguous']==1]
 return {'accuracy':acc,'average_support_used':sum(r['support_used'] for r in rows)/n,'support_used_distribution':dist,'accuracy_by_support_used':by,'per_family_accuracy':pf,'collision_case_accuracy':sum(r['correct'] for r in col)/max(1,len(col)),'ambiguous_case_accuracy':sum(r['correct'] for r in amb)/max(1,len(amb)),'error_count':sum(1-r['correct'] for r in rows)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='9251,9252,9253,9254,9255'); ap.add_argument('--train-rows-per-seed',type=int,default=1000); ap.add_argument('--test-rows-per-seed',type=int,default=1000); ap.add_argument('--ood-rows-per-seed',type=int,default=1000); ap.add_argument('--max-support-count',type=int,default=5); ap.add_argument('--workers',default='auto'); ap.add_argument('--cpu-target',default='saturate'); ap.add_argument('--heartbeat-sec',type=int,default=20); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 cpath=Path('target/pilot_wave/d44c_adaptive_support_expansion_prototype/smoke')
 if not cpath.exists(): raise SystemExit('missing D44C artifacts')
 dec=json.loads((cpath/'decision.json').read_text()); aggc=json.loads((cpath/'aggregate_metrics.json').read_text()); apol=json.loads((cpath/'adaptive_policy_report.json').read_text()); seff=json.loads((cpath/'support_efficiency_report.json').read_text())
 (out/'d44c_upstream_manifest.json').write_text(json.dumps({'decision':dec,'has_aggregate':True},indent=2))
 audit={
  'ambiguity_detection_precision_definition':'in D44C code it was set to ambiguous_case_accuracy (not precision over TP/FP).',
  'ambiguity_detection_recall_definition':'in D44C code it was set to 1-accuracy_by_support_used[1] proxy (not true recall).',
  'support_request_precision_definition':'proxy 1-(avg_support-1)/4.',
  'support_request_recall_definition':'proxy min(1,(avg_support-1)/2).',
  'unresolved_after_max_support_rate_definition':'taken from conservative abstain_rate, not soft arm unresolved.',
  'unresolved_after_max_support_rate_arm':'conservative',
  'why_soft_1_0_with_unresolved_0_3836':'soft arm always predicts after escalation; unresolved metric came from conservative abstain behavior.',
  'metric_names_misleading':True,
  'metric_definitions_to_change':['ambiguous_case','support_request_needed','support_request_made','unresolved_after_max per arm']
 }
 (out/'d44c_metric_semantics_audit.json').write_text(json.dumps(audit,indent=2))
 (out/'dataset_manifest.json').write_text(json.dumps({'families':F,'pairs':PAIRS,'supports':[1,2,3,5]},indent=2))
 (out/'policy_definitions_report.json').write_text(json.dumps({'arms':ARMS},indent=2))
 seeds=[int(x) for x in a.seeds.split(',') if x]
 mets={arm:{'train':[],'test':[],'ood':[]} for arm in ARMS}; testrows={arm:[] for arm in ARMS}
 for s in seeds:
  rng=random.Random(s)
  def mk(n,ood=False):
   return [{'truth':(fam:=rng.choice(F)),'supports':[gen(rng,fam,ood) for _ in range(5)]} for _ in range(n)]
  tr,mte,od=mk(a.train_rows_per_seed,False),mk(a.test_rows_per_seed,False),mk(a.ood_rows_per_seed,True)
  for arm in ARMS:
   for split,ds in [('train',tr),('test',mte),('ood',od)]:
    rows=[run_row(r,arm,rng) for r in ds]
    sm=summarize(rows); mets[arm][split].append(sm)
    if split=='test': testrows[arm].extend(rows)
 agg={'arms':{},'failed_jobs':0}
 per_family={}; per_support={}; coll_amb={}
 for arm in ARMS:
  t=statistics.mean(x['accuracy'] for x in mets[arm]['test']); o=statistics.mean(x['accuracy'] for x in mets[arm]['ood']); tr=statistics.mean(x['accuracy'] for x in mets[arm]['train'])
  avg=statistics.mean(x['average_support_used'] for x in mets[arm]['test'])
  row=mets[arm]['test'][0]
  agg['arms'][arm]={'train_accuracy':tr,'test_accuracy':t,'OOD_accuracy':o,'average_support_used':avg,'support_used_distribution':row['support_used_distribution'],'accuracy_by_support_used':row['accuracy_by_support_used'],'per_family_accuracy':{f:statistics.mean(x['per_family_accuracy'][f] for x in mets[arm]['test']) for f in F},'collision_case_accuracy':statistics.mean(x['collision_case_accuracy'] for x in mets[arm]['test']),'ambiguous_case_accuracy':statistics.mean(x['ambiguous_case_accuracy'] for x in mets[arm]['test']),'abstain_rate':0.0,'effective_accuracy_counting_abstain_wrong':t,'error_count':statistics.mean(x['error_count'] for x in mets[arm]['test']),'failed_seed_count':0}
  per_family[arm]=agg['arms'][arm]['per_family_accuracy']; per_support[arm]=agg['arms'][arm]['accuracy_by_support_used']; coll_amb[arm]={'collision_case_accuracy':agg['arms'][arm]['collision_case_accuracy'],'ambiguous_case_accuracy':agg['arms'][arm]['ambiguous_case_accuracy']}
 # adaptive metrics repaired
 old=agg['arms']['OLD_ADAPTIVE_1_TO_5_REPLAY']; staged=agg['arms']['STAGED_SUPPORT_1_TO_2_TO_3_TO_5']; rnd=agg['arms']['RANDOM_EXTRA_SUPPORT_CONTROL']; one=agg['arms']['ONE_SHOT_SUPPORT_1_BASELINE']; fix5=agg['arms']['FIXED_SUPPORT_5']; oracle=agg['arms']['ORACLE_MINIMAL_SUPPORT_UPPER_BOUND']
 def req_metrics(rows):
  need=[(r['ambiguous']==1 or r['correct']==0) for r in rows]; made=[r['support_used']>1 for r in rows]
  tp=sum(1 for n,m in zip(need,made) if n and m); fp=sum(1 for n,m in zip(need,made) if (not n) and m); fn=sum(1 for n,m in zip(need,made) if n and (not m))
  prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
  return prec,rec,fp/max(1,len(rows)),fn/max(1,len(rows))
 sprec,srec,unnec,miss=req_metrics(testrows['STAGED_SUPPORT_1_TO_2_TO_3_TO_5'])
 support_eff={'support_request_precision':sprec,'support_request_recall':srec,'unnecessary_extra_support_rate':unnec,'missed_extra_support_rate':miss,'unresolved_after_max_support_rate':sum(1 for r in testrows['STAGED_SUPPORT_1_TO_2_TO_3_TO_5'] if r['support_used']==5 and r['ambiguous']==1)/max(1,len(testrows['STAGED_SUPPORT_1_TO_2_TO_3_TO_5'])),'average_support_saved_vs_fixed_5':5-staged['average_support_used'],'average_support_saved_vs_old_adaptive':old['average_support_used']-staged['average_support_used'],'accuracy_gain_vs_support_1':staged['test_accuracy']-one['test_accuracy'],'accuracy_gain_vs_random_extra_support':staged['test_accuracy']-rnd['test_accuracy'],'accuracy_loss_vs_fixed_5':fix5['test_accuracy']-staged['test_accuracy'],'accuracy_loss_vs_old_adaptive':old['test_accuracy']-staged['test_accuracy'],'oracle_minimal_average_support':oracle['average_support_used'],'gap_to_oracle_minimal_support':staged['average_support_used']-oracle['average_support_used']}
 # write many reports
 repmap={'one_shot_support_1_report.json':'ONE_SHOT_SUPPORT_1_BASELINE','fixed_support_2_report.json':'FIXED_SUPPORT_2','fixed_support_3_report.json':'FIXED_SUPPORT_3','fixed_support_5_report.json':'FIXED_SUPPORT_5','old_adaptive_1_to_5_replay_report.json':'OLD_ADAPTIVE_1_TO_5_REPLAY','staged_support_1_to_2_to_3_to_5_report.json':'STAGED_SUPPORT_1_TO_2_TO_3_TO_5','staged_support_margin_policy_report.json':'STAGED_SUPPORT_MARGIN_POLICY','staged_support_entropy_policy_report.json':'STAGED_SUPPORT_ENTROPY_POLICY','staged_support_hybrid_policy_report.json':'STAGED_SUPPORT_HYBRID_POLICY','random_extra_support_control_report.json':'RANDOM_EXTRA_SUPPORT_CONTROL','bad_ambiguity_signal_control_report.json':'BAD_AMBIGUITY_SIGNAL_CONTROL','oracle_minimal_support_upper_bound_report.json':'ORACLE_MINIMAL_SUPPORT_UPPER_BOUND'}
 for f,a2 in repmap.items(): (out/f).write_text(json.dumps(agg['arms'][a2],indent=2))
 (out/'ambiguity_metric_repair_report.json').write_text(json.dumps({'definitions':{'ambiguous_case':'sample where one-support is not fairly identifiable','support_request_needed':'sample where one-shot is wrong or ambiguous','support_request_made':'policy used >1 support','unresolved_after_max':'still ambiguous at max support for that arm only'}},indent=2))
 (out/'support_efficiency_report.json').write_text(json.dumps(support_eff,indent=2))
 frontier={k:{'test_accuracy':agg['arms'][k]['test_accuracy'],'average_support_used':agg['arms'][k]['average_support_used']} for k in ARMS}
 (out/'accuracy_efficiency_frontier_report.json').write_text(json.dumps(frontier,indent=2))
 (out/'per_family_accuracy_report.json').write_text(json.dumps(per_family,indent=2)); (out/'per_support_count_accuracy_report.json').write_text(json.dumps(per_support,indent=2)); (out/'collision_ambiguous_case_report.json').write_text(json.dumps(coll_amb,indent=2));
 (out/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2))
 dec='old_adaptive_policy_remains_best'; ver='D44C2_OLD_ADAPTIVE_REMAINS_BEST'; nxt='D44D_PRIMITIVE_SPACE_REDESIGN_PLAN'
 if staged['test_accuracy']>=0.999 and staged['OOD_accuracy']>=0.999 and staged['average_support_used']<=2.20 and support_eff['accuracy_loss_vs_old_adaptive']<=0.001 and support_eff['average_support_saved_vs_old_adaptive']>=0.20 and support_eff['average_support_saved_vs_fixed_5']>=2.70:
  dec='staged_adaptive_support_policy_optimized'; ver='D44C2_STAGED_SUPPORT_POLICY_OPTIMIZED'
 elif staged['test_accuracy']>=0.999 and staged['OOD_accuracy']>=0.999:
  dec='adaptive_support_accuracy_confirmed_no_efficiency_gain'; ver='D44C2_ACCURACY_NO_EFFICIENCY_GAIN'
 elif staged['test_accuracy']<old['test_accuracy']:
  dec='staged_support_policy_not_confirmed'; ver='D44C2_STAGED_POLICY_NOT_CONFIRMED'; nxt='D44C3_SUPPORT_POLICY_DIAGNOSTIC'
 (out/'decision.json').write_text(json.dumps({'decision':dec,'verdict':ver,'next':nxt},indent=2)); (out/'summary.json').write_text(json.dumps({'decision':dec,'next':nxt},indent=2)); (out/'report.md').write_text('D44C2 staged support policy and metric audit only.\n')
if __name__=='__main__': main()
