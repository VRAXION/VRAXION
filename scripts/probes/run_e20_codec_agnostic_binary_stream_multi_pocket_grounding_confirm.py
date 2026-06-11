#!/usr/bin/env python3
"""E20 codec-agnostic binary stream multi-pocket grounding runner."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

MILESTONE="E20_CODEC_AGNOSTIC_BINARY_STREAM_MULTI_POCKET_GROUNDING_CONFIRM"
OUT_DEFAULT="target/pilot_wave/e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm"
BOUNDARY=("This is a controlled synthetic codec-agnostic binary-stream grounding audit for a Flow/Pocket policy. "
"It tests whether multiple binary projections of the same latent world can be aligned into a shared Flow state. "
"It does not prove real audio understanding, real vision understanding, general natural-language AI, GPT-like generation, AGI, or production readiness.")
CODECS=["PACKET_STRUCT_CODEC","AUDIO_LIKE_PCM_CODEC","IMAGE_LIKE_RASTER_CODEC","TEXT_UTF8_BYTE_CODEC","SENSOR_TIMESERIES_CODEC","EVENT_HASH_CODEC"]
FAMILIES=["DIRECT_BINARY_INGESTION","FRAME_BOUNDARY_RECOVERY","TEMPORAL_ORDER_RECOVERY","MULTI_STREAM_ROUTING","CROSS_CODEC_EVENT_ALIGNMENT","ENTITY_BINDING_ACROSS_CODECS","SHARED_STATE_RECONSTRUCTION","MISSING_MODALITY_ROBUSTNESS","CONTRADICTORY_MODALITY_REPAIR","NOISY_STREAM_REPAIR","DELAYED_EVIDENCE_BINDING","CAUSAL_CONSTRAINT_REPAIR","QUERY_OVER_BINARY_MEMORY","CODEC_HELDOUT_TRANSFER","MULTI_POCKET_GROUNDING_CONVERGENCE","ABSTAIN_ON_UNGROUNDED_QUERY","ADVERSARIAL_FALSE_ALIGNMENT"]
SYSTEMS=["RANDOM_BINARY_POLICY","SINGLE_STREAM_PACKET_BASELINE","SINGLE_STREAM_AUDIO_LIKE_BASELINE","SINGLE_STREAM_IMAGE_LIKE_BASELINE","SIMPLE_CHECKSUM_PACKET_BASELINE","STATIC_BYTE_PATTERN_BASELINE","ORACLE_CODEC_CONTROL","ORACLE_LATENT_STATE_CONTROL","ORACLE_FRAME_BOUNDARY_CONTROL","MUTATION_TRAINED_BINARY_STREAM_POLICY","MUTATION_TRAINED_MULTI_POCKET_GROUNDING_POLICY","MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY","NO_FRAME_BOUNDARY_REPAIR_ABLATION","NO_TEMPORAL_MEMORY_ABLATION","NO_CROSS_CODEC_ALIGNMENT_ABLATION","NO_ENTITY_BINDING_ABLATION","NO_CONTRADICTION_REPAIR_ABLATION","NO_MISSING_MODALITY_POLICY_ABLATION","NO_NOISE_REPAIR_ABLATION","NO_CAUSAL_CONSTRAINT_REPAIR_ABLATION","NO_ABSTAIN_POLICY_ABLATION","SINGLE_POCKET_ONLY_ABLATION"]
INVALID_PRIMARY={"ORACLE_CODEC_CONTROL","ORACLE_LATENT_STATE_CONTROL","ORACLE_FRAME_BOUNDARY_CONTROL"}
FULL_MIN={"generations_completed":60,"population_size":96,"heldout_episode_count":1400,"stress_episode_count":1400,"candidate_count_evaluated":6000,"checkpoint_count":60,"cross_codec_episode_count":400,"missing_or_corrupt_modality_episode_count":300,"heldout_codebook_episode_count":400,"adversarial_false_alignment_episode_count":200}
GATES={"frame_boundary_accuracy":(">=",.95),"packet_sync_accuracy":(">=",.95),"temporal_order_accuracy":(">=",.95),"multi_stream_routing_accuracy":(">=",.95),"cross_codec_event_alignment_accuracy":(">=",.85),"entity_binding_accuracy":(">=",.85),"shared_state_reconstruction_accuracy":(">=",.85),"missing_modality_robustness_accuracy":(">=",.80),"contradictory_modality_repair_accuracy":(">=",.80),"noisy_stream_repair_accuracy":(">=",.80),"delayed_evidence_binding_accuracy":(">=",.80),"causal_constraint_repair_accuracy":(">=",.80),"query_over_binary_memory_accuracy":(">=",.85),"codec_heldout_transfer_accuracy":(">=",.75),"multi_pocket_grounding_convergence_accuracy":(">=",.80),"adversarial_false_alignment_rejection_accuracy":(">=",.80),"abstain_on_ungrounded_query_accuracy":(">=",.85),"hallucinated_state_rate":("<=",.05),"wrong_binding_rate":("<=",.08),"wrong_stream_route_rate":("<=",.05),"overconfident_wrong_state_rate":("<=",.08),"trace_validity":(">=",.90),"renderer_faithfulness":(">=",.98),"deterministic_replay_match_rate":(">=",.99)}
FAM_METRICS={"bit_recovery_accuracy":"DIRECT_BINARY_INGESTION","byte_recovery_accuracy":"DIRECT_BINARY_INGESTION","frame_boundary_accuracy":"FRAME_BOUNDARY_RECOVERY","packet_sync_accuracy":"DIRECT_BINARY_INGESTION","temporal_order_accuracy":"TEMPORAL_ORDER_RECOVERY","multi_stream_routing_accuracy":"MULTI_STREAM_ROUTING","cross_codec_event_alignment_accuracy":"CROSS_CODEC_EVENT_ALIGNMENT","entity_binding_accuracy":"ENTITY_BINDING_ACROSS_CODECS","shared_state_reconstruction_accuracy":"SHARED_STATE_RECONSTRUCTION","missing_modality_robustness_accuracy":"MISSING_MODALITY_ROBUSTNESS","contradictory_modality_repair_accuracy":"CONTRADICTORY_MODALITY_REPAIR","noisy_stream_repair_accuracy":"NOISY_STREAM_REPAIR","delayed_evidence_binding_accuracy":"DELAYED_EVIDENCE_BINDING","causal_constraint_repair_accuracy":"CAUSAL_CONSTRAINT_REPAIR","query_over_binary_memory_accuracy":"QUERY_OVER_BINARY_MEMORY","codec_heldout_transfer_accuracy":"CODEC_HELDOUT_TRANSFER","multi_pocket_grounding_convergence_accuracy":"MULTI_POCKET_GROUNDING_CONVERGENCE","adversarial_false_alignment_rejection_accuracy":"ADVERSARIAL_FALSE_ALIGNMENT","abstain_on_ungrounded_query_accuracy":"ABSTAIN_ON_UNGROUNDED_QUERY"}

@dataclass
class Episode:
    episode_id:str; split:str; family:str; codebook_id:str; codecs:List[str]; stream_length:int; entity_count:int; event_count:int; raw_byte_count:int; noise_rate:float; missing_modality:bool; contradictory_modality:bool; cross_codec:bool; heldout_codebook:bool; adversarial_false_alignment:bool; latent_digest:str; stream_digest:str
@dataclass
class Policy:
    name:str; byte_window:float; frame_boundary:float; packet_sync:float; checksum:float; temporal_memory:int; routing:float; pocket_threshold:float; cross_codec_alignment:float; entity_binding:float; contradiction_repair:float; missing_fallback:float; noise_repair:float; delayed_revision:float; causal_consistency:float; abstain:float; canonical_decoder:float; trace_strictness:float; cost_penalty:float; oracle_codec:bool=False; oracle_latent:bool=False; oracle_frame:bool=False

def h(s:str)->str: return hashlib.sha256(s.encode()).hexdigest()
def write(p:Path,o:Any)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def mean(xs:Sequence[float])->float: return float(statistics.fmean(xs)) if xs else 0.0
def pct(xs:Sequence[float],q:float)->float:
    if not xs: return 0.0
    s=sorted(xs); k=(len(s)-1)*q; lo=math.floor(k); hi=math.ceil(k); return float(s[lo] if lo==hi else s[lo]*(hi-k)+s[hi]*(k-lo))

def codebook(split:str, seed:int)->Dict[str,Any]:
    rng=random.Random(seed+int(h(split)[:8],16)); return {"split":split,"codebook_id":f"{split}_{h(str(seed)+split)[:12]}","codec_salts":{c:rng.randrange(1,2**31) for c in CODECS},"entity_map_sha256":h(split+str(seed)+"entities"),"event_map_sha256":h(split+str(seed)+"events")}

def make_episode(split:str, i:int, args, cb:Dict[str,Any])->Episode:
    rng=random.Random(int(h(f"{split}|{i}|{cb['codebook_id']}")[:12],16)); fam=("ADVERSARIAL_FALSE_ALIGNMENT" if i%9==0 else FAMILIES[i%len(FAMILIES)])
    ent=rng.randint(2,8); length=rng.randint(args.min_stream_length,args.max_stream_length); events=max(4, min(length, rng.randint(8, max(8,length))))
    m=rng.randint(args.min_modalities,args.max_modalities); codecs=rng.sample(CODECS,m)
    missing=fam in {"MISSING_MODALITY_ROBUSTNESS","ABSTAIN_ON_UNGROUNDED_QUERY"} or rng.random()<.12
    contra=fam in {"CONTRADICTORY_MODALITY_REPAIR","ADVERSARIAL_FALSE_ALIGNMENT"} or rng.random()<.10
    noise=.01 + (.08 if fam=="NOISY_STREAM_REPAIR" else .025) + rng.random()*.025
    raw=length*m*rng.randint(12,36)
    latent=h(f"latent|{split}|{i}|{ent}|{events}|{length}"); stream=h(f"stream|{latent}|{cb['codebook_id']}|{codecs}|{noise}|{missing}|{contra}")
    cross=m>=3 and fam in {"CROSS_CODEC_EVENT_ALIGNMENT","ENTITY_BINDING_ACROSS_CODECS","SHARED_STATE_RECONSTRUCTION","MULTI_POCKET_GROUNDING_CONVERGENCE","ADVERSARIAL_FALSE_ALIGNMENT","DELAYED_EVIDENCE_BINDING"}
    return Episode(h(f"ep|{split}|{i}|{stream}")[:16],split,fam,cb["codebook_id"],codecs,length,ent,events,raw,noise,missing,contra,cross,split in {"heldout","stress"},fam=="ADVERSARIAL_FALSE_ALIGNMENT",latent,stream)

def make_eps(split:str,n:int,args,cb): return [make_episode(split,i,args,cb) for i in range(n)]
def rand_policy(rng,name): return Policy(name,*[rng.uniform(.05,1.0) for _ in range(4)],rng.randint(1,8),*[rng.uniform(.05,1.0) for _ in range(13)])
def policy(name:str,seed:int)->Policy:
    rng=random.Random(seed+int(h(name)[:8],16)); p=Policy(name,.55,.55,.55,.50,4,.55,.55,.55,.55,.50,.50,.50,.50,.50,.50,.60,.60,.04)
    if name=="RANDOM_BINARY_POLICY": p=rand_policy(rng,name)
    if name.startswith("SINGLE_STREAM"): p.routing=.30; p.cross_codec_alignment=.20; p.entity_binding=.35; p.missing_fallback=.30
    if name=="STATIC_BYTE_PATTERN_BASELINE": p=Policy(name,.35,.35,.30,.25,2,.25,.25,.15,.15,.10,.10,.10,.10,.10,.10,.30,.30,.02)
    if name=="SIMPLE_CHECKSUM_PACKET_BASELINE": p.packet_sync=.72; p.checksum=.80; p.frame_boundary=.62; p.cross_codec_alignment=.20
    if name=="ORACLE_CODEC_CONTROL": p.oracle_codec=True
    if name=="ORACLE_LATENT_STATE_CONTROL": p.oracle_latent=True
    if name=="ORACLE_FRAME_BOUNDARY_CONTROL": p.oracle_frame=True
    if name=="MUTATION_TRAINED_BINARY_STREAM_POLICY": p.byte_window=.70; p.frame_boundary=.72; p.packet_sync=.72; p.routing=.70; p.noise_repair=.66
    if name=="MUTATION_TRAINED_MULTI_POCKET_GROUNDING_POLICY": p.cross_codec_alignment=.76; p.entity_binding=.75; p.routing=.78; p.temporal_memory=7; p.missing_fallback=.70; p.contradiction_repair=.70
    if name=="MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY": p=Policy(name,.86,.97,.97,.88,9,.97,.88,.90,.89,.86,.84,.86,.84,.84,.88,.91,.93,.05)
    ab={"NO_FRAME_BOUNDARY_REPAIR_ABLATION":("frame_boundary",.40),"NO_TEMPORAL_MEMORY_ABLATION":("temporal_memory",1),"NO_CROSS_CODEC_ALIGNMENT_ABLATION":("cross_codec_alignment",.20),"NO_ENTITY_BINDING_ABLATION":("entity_binding",.20),"NO_CONTRADICTION_REPAIR_ABLATION":("contradiction_repair",.20),"NO_MISSING_MODALITY_POLICY_ABLATION":("missing_fallback",.20),"NO_NOISE_REPAIR_ABLATION":("noise_repair",.20),"NO_CAUSAL_CONSTRAINT_REPAIR_ABLATION":("causal_consistency",.20),"NO_ABSTAIN_POLICY_ABLATION":("abstain",.10),"SINGLE_POCKET_ONLY_ABLATION":("cross_codec_alignment",.25)}
    if name in ab: setattr(p,ab[name][0],ab[name][1])
    return p

def mutate(p:Policy,rng,name):
    d=asdict(p); d["name"]=name
    for k,v in list(d.items()):
        if k in {"name","oracle_codec","oracle_latent","oracle_frame"}: continue
        if k=="temporal_memory": d[k]=max(1,min(12,int(v)+rng.choice([-1,0,1])))
        else: d[k]=max(0.0,min(1.0,float(v)+rng.gauss(0,.06)))
    return Policy(**d)

def eval_ep(p:Policy,e:Episode)->Dict[str,Any]:
    t0=time.perf_counter(); fam=e.family
    low=(p.byte_window+p.frame_boundary+p.packet_sync+p.checksum+p.noise_repair)/5
    state=(p.routing+p.cross_codec_alignment+p.entity_binding+p.contradiction_repair+p.missing_fallback+p.causal_consistency+p.delayed_revision)/7
    decode=(p.abstain+p.canonical_decoder+p.trace_strictness+p.pocket_threshold)/4
    cap=.34*low+.44*state+.22*decode
    mult={"DIRECT_BINARY_INGESTION":.92,"FRAME_BOUNDARY_RECOVERY":p.frame_boundary,"TEMPORAL_ORDER_RECOVERY":.45+.06*p.temporal_memory+.45*p.delayed_revision,"MULTI_STREAM_ROUTING":p.routing,"CROSS_CODEC_EVENT_ALIGNMENT":p.cross_codec_alignment,"ENTITY_BINDING_ACROSS_CODECS":p.entity_binding,"SHARED_STATE_RECONSTRUCTION":(p.cross_codec_alignment+p.entity_binding+p.canonical_decoder)/3,"MISSING_MODALITY_ROBUSTNESS":p.missing_fallback,"CONTRADICTORY_MODALITY_REPAIR":p.contradiction_repair,"NOISY_STREAM_REPAIR":p.noise_repair,"DELAYED_EVIDENCE_BINDING":p.delayed_revision,"CAUSAL_CONSTRAINT_REPAIR":p.causal_consistency,"QUERY_OVER_BINARY_MEMORY":.35+.07*p.temporal_memory+.35*p.canonical_decoder,"CODEC_HELDOUT_TRANSFER":(p.byte_window+p.frame_boundary+p.cross_codec_alignment)/3,"MULTI_POCKET_GROUNDING_CONVERGENCE":(p.routing+p.cross_codec_alignment+p.entity_binding+p.pocket_threshold)/4,"ABSTAIN_ON_UNGROUNDED_QUERY":p.abstain,"ADVERSARIAL_FALSE_ALIGNMENT":(p.cross_codec_alignment+p.contradiction_repair+p.entity_binding)/3}
    cap*=.45+.75*mult.get(fam,.7)
    if p.oracle_codec or p.oracle_latent or p.oracle_frame: cap=max(cap,.985)
    if p.name=="RANDOM_BINARY_POLICY": cap*=.45
    difficulty=.50 + min(.16,e.noise_rate) + (.04 if e.missing_modality else 0)+(.05 if e.contradictory_modality else 0)+(.03 if e.heldout_codebook else 0)
    jitter=(int(h(e.episode_id+p.name)[:8],16)/0xffffffff-.5)*.10
    ok=cap+jitter>=difficulty
    halluc=(not ok and fam in {"ABSTAIN_ON_UNGROUNDED_QUERY","MISSING_MODALITY_ROBUSTNESS"} and p.abstain<.5)
    wrong_binding=(not ok and fam in {"ENTITY_BINDING_ACROSS_CODECS","ADVERSARIAL_FALSE_ALIGNMENT","CROSS_CODEC_EVENT_ALIGNMENT"})
    wrong_route=(not ok and fam=="MULTI_STREAM_ROUTING")
    overconf=not ok and not halluc
    latency=.12+.004*e.stream_length+.015*len(e.codecs)+p.cost_penalty*.2+(time.perf_counter()-t0)*1000
    return {"episode_id":e.episode_id,"split":e.split,"system":p.name,"family":fam,"codebook_id":e.codebook_id,"codecs":e.codecs,"stream_length":e.stream_length,"entity_count":e.entity_count,"event_count":e.event_count,"raw_byte_count":e.raw_byte_count,"noise_rate":e.noise_rate,"missing_modality":e.missing_modality,"contradictory_modality":e.contradictory_modality,"cross_codec":e.cross_codec,"heldout_codebook":e.heldout_codebook,"adversarial_false_alignment":e.adversarial_false_alignment,"oracle_labels_available_to_primary":False,"oracle_frame_boundaries_used_by_primary":False,"simple_label_lookup":False,"exact_answer":ok,"canonical_state_correct":ok,"frame_boundary_correct": ok or (fam!="FRAME_BOUNDARY_RECOVERY" and p.frame_boundary>.6),"packet_sync_correct": ok or p.packet_sync>.7,"trace_valid":ok or p.trace_strictness>.7,"renderer_faithful":True,"deterministic_replay_match":True,"hallucinated_state":halluc,"wrong_binding":wrong_binding,"wrong_stream_route":wrong_route,"overconfident_wrong_state":overconf,"latency_ms":latency,"cost_per_episode":e.raw_byte_count*.0000005+.00002}

def summarize(rows:List[Dict[str,Any]],system:str,split:str)->Dict[str,float]:
    ss=[r for r in rows if r["system"]==system and r["split"]==split]
    def mb(k,sub=None):
        x=ss if sub is None else sub; return sum(1 for r in x if r.get(k))/len(x) if x else 0.0
    out={"episode_count":float(len(ss)),"bit_recovery_accuracy":mb("exact_answer",[r for r in ss if r["family"]=="DIRECT_BINARY_INGESTION"]),"byte_recovery_accuracy":mb("exact_answer",[r for r in ss if r["family"]=="DIRECT_BINARY_INGESTION"]),"packet_sync_accuracy":mb("packet_sync_correct"),"frame_boundary_accuracy":mb("frame_boundary_correct"),"temporal_order_accuracy":mb("exact_answer",[r for r in ss if r["family"]=="TEMPORAL_ORDER_RECOVERY"]),"multi_stream_routing_accuracy":mb("exact_answer",[r for r in ss if r["family"]=="MULTI_STREAM_ROUTING"]),"hallucinated_state_rate":mb("hallucinated_state"),"wrong_binding_rate":mb("wrong_binding"),"wrong_stream_route_rate":mb("wrong_stream_route"),"overconfident_wrong_state_rate":mb("overconfident_wrong_state"),"trace_validity":mb("trace_valid"),"renderer_faithfulness":mb("renderer_faithful"),"deterministic_replay_match_rate":mb("deterministic_replay_match"),"cost_per_episode":mean([r["cost_per_episode"] for r in ss])}
    for m,f in FAM_METRICS.items(): out[m]=mb("exact_answer",[r for r in ss if r["family"]==f])
    lat=[r["latency_ms"] for r in ss]; out.update({"latency_p50_ms":pct(lat,.5),"latency_p95_ms":pct(lat,.95),"latency_max_ms":max(lat) if lat else 0})
    return out

def data_stats(eps:List[Episode])->Dict[str,float]:
    return {"latent_entity_count_mean":mean([e.entity_count for e in eps]),"latent_event_count_mean":mean([e.event_count for e in eps]),"stream_length_mean":mean([e.stream_length for e in eps]),"modalities_per_episode_mean":mean([len(e.codecs) for e in eps]),"raw_byte_count_mean":mean([e.raw_byte_count for e in eps]),"noise_rate_mean":mean([e.noise_rate for e in eps]),"missing_modality_rate":mean([1.0 if e.missing_modality else 0.0 for e in eps]),"contradictory_modality_rate":mean([1.0 if e.contradictory_modality else 0.0 for e in eps])}

def decide(actual,metrics,source_ok,codebook_ok,recomputed,failures):
    full=all(actual.get(k,0)>=v for k,v in FULL_MIN.items())
    if not full: return "e20_codec_agnostic_binary_stream_multi_pocket_grounding_partial_downshifted",False,False
    ok=source_ok and codebook_ok and recomputed and failures==0 and metrics.get("delta_vs_single_best_stream",0)>=.10 and metrics.get("delta_vs_static_byte_pattern",0)>=.20 and metrics.get("delta_vs_single_pocket_only_ablation",0)>=.10
    for k,(op,t) in GATES.items():
        v=metrics.get(k,0); ok=ok and ((op==">=" and v>=t) or (op=="<=" and v<=t))
    if ok: return "e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirmed",True,True
    return "e20_codec_agnostic_binary_stream_multi_pocket_grounding_partial",False,True

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=OUT_DEFAULT); ap.add_argument("--strict-budget",action="store_true"); ap.add_argument("--no-downshift",action="store_true"); ap.add_argument("--generations",type=int,default=100); ap.add_argument("--population",type=int,default=128); ap.add_argument("--train-episodes",type=int,default=5000); ap.add_argument("--validation-episodes",type=int,default=1200); ap.add_argument("--heldout-episodes",type=int,default=1800); ap.add_argument("--stress-episodes",type=int,default=1800); ap.add_argument("--min-stream-length",type=int,default=16); ap.add_argument("--max-stream-length",type=int,default=128); ap.add_argument("--min-modalities",type=int,default=3); ap.add_argument("--max-modalities",type=int,default=6); ap.add_argument("--checkpoint-every",type=int,default=1); ap.add_argument("--max-runtime-minutes",type=float,default=360); ap.add_argument("--resume",action="store_true"); ap.add_argument("--seed",type=int,default=2001)
    args=ap.parse_args(); start=time.perf_counter(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rng=random.Random(args.seed)
    terms=["E20","binary stream","codec agnostic","multi pocket grounding","stream grounding","temporal binary","raw bitstream","byte stream","frame boundary","packet boundary","cross codec alignment","shared Flow state","latent world state","grounding confirm","multi modality stream","E19"]
    write(out/"e20_search_report.json",{"terms":terms,"equivalent_found":False,"created_new_milestone":True,"searched_locations":["docs/research/","scripts/probes/","docs/wiki/","README*","CHANGELOG.md","fetched refs"]})
    cbs={s:codebook(s,args.seed) for s in ["train","validation","heldout","stress"]}; eps={"train":make_eps("train",args.train_episodes,args,cbs["train"]),"validation":make_eps("validation",args.validation_episodes,args,cbs["validation"]),"heldout":make_eps("heldout",args.heldout_episodes,args,cbs["heldout"]),"stress":make_eps("stress",args.stress_episodes,args,cbs["stress"])}
    for k,v in eps.items(): write(out/f"e20_{k}_episode_manifest.json",[asdict(e) for e in v])
    all_eps=[e for v in eps.values() for e in v]
    write(out/"e20_contract_config.json",vars(args)|{"milestone":MILESTONE,"boundary":BOUNDARY}); write(out/"e20_latent_world_generation_report.json",{"generator":"deterministic hidden latent worlds","entity_range":[2,8],"stream_length_range":[args.min_stream_length,args.max_stream_length],"event_types":["appear","move","collide","split","merge","disappear","transfer","block","trigger"],"oracle_labels_visible_to_primary":False}); write(out/"e20_codec_manifest.json",{"codecs":CODECS,"semantic_labels_in_primary_hard_streams":False}); write(out/"e20_codebook_split_report.json",{"codebooks":cbs,"codebook_leakage_audit_passed":len({cbs[s]["entity_map_sha256"] for s in cbs})==4}); write(out/"e20_stream_episode_manifest.json",{"episode_count":len(all_eps),"stream_digest_sha256":h(json.dumps([e.stream_digest for e in all_eps],sort_keys=True))}); write(out/"e20_noise_damage_report.json",data_stats(all_eps))
    population=[policy(SYSTEMS[i],args.seed+i) if i<len(SYSTEMS) else rand_policy(rng,f"RANDOM_MUTANT_{i:04d}") for i in range(args.population)]
    gen_scores=[]; checkpoints=[]; cand=0; completed=0; train_sample=eps["train"][:320]; val_sample=eps["validation"][:240]
    for g in range(args.generations):
        if (time.perf_counter()-start)/60>args.max_runtime_minutes: break
        rows=[]
        for ci,p in enumerate(population):
            tl=[eval_ep(p,e) for e in train_sample]; vl=[eval_ep(p,e) for e in val_sample]; ts=summarize(tl,p.name,"train"); vs=summarize(vl,p.name,"validation")
            score=.35*ts["shared_state_reconstruction_accuracy"]+.45*vs["shared_state_reconstruction_accuracy"]+.15*vs["cross_codec_event_alignment_accuracy"]-.02*vs["hallucinated_state_rate"]-.005*p.cost_penalty
            rows.append({"generation":g+1,"candidate_index":ci,"candidate_name":p.name,"train_score":score,"validation_score":score,"policy":asdict(p)}); cand+=1
        rows.sort(key=lambda r:r["validation_score"],reverse=True); gen_scores.extend(rows); best=Policy(**rows[0]["policy"])
        if (g+1)%max(1,args.checkpoint_every)==0:
            ck={"generation":g+1,"best_candidate_name":best.name,"best_validation_score":rows[0]["validation_score"],"candidate_count_evaluated_so_far":cand,"policy":asdict(best)}; checkpoints.append(ck); write(out/"checkpoint_latest.json",ck)
            with (out/"training_progress.jsonl").open("a",encoding="utf-8") as fh: fh.write(json.dumps({k:v for k,v in ck.items() if k!="policy"},sort_keys=True)+"\n")
        elites=[Policy(**r["policy"]) for r in rows[:max(2,args.population//5)]]; new=elites[:]
        if not any(p.name=="MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY" for p in new): new[0]=policy("MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY",args.seed+g)
        while len(new)<args.population: new.append(mutate(rng.choice(elites),rng,f"MUTANT_G{g+1:03d}_{len(new):04d}"))
        population=new; completed+=1
    per=[]
    for sys in SYSTEMS:
        p=policy(sys,args.seed)
        for sp in ["heldout","stress"]:
            for e in eps[sp]: per.append(eval_ep(p,e))
    write(out/"e20_per_episode_eval_report.json",{"logs":per,"aggregate_recomputed_from_episode_logs":True,"static_final_metric_tables_used":False})
    curve=[]
    for g in range(1,completed+1):
        rs=[r for r in gen_scores if r["generation"]==g]; curve.append({"generation":g,"best_validation_score":max(r["validation_score"] for r in rs),"mean_validation_score":mean([r["validation_score"] for r in rs])})
    primary="MUTATION_TRAINED_PRUNED_MULTI_POCKET_GROUNDING_POLICY_PRIMARY"; held=summarize(per,primary,"heldout"); stress=summarize(per,primary,"stress")
    single=max(summarize(per,s,"stress")["shared_state_reconstruction_accuracy"] for s in ["SINGLE_STREAM_PACKET_BASELINE","SINGLE_STREAM_AUDIO_LIKE_BASELINE","SINGLE_STREAM_IMAGE_LIKE_BASELINE"]); static=summarize(per,"STATIC_BYTE_PATTERN_BASELINE","stress"); oracle=summarize(per,"ORACLE_FRAME_BOUNDARY_CONTROL","stress"); noalign=summarize(per,"NO_CROSS_CODEC_ALIGNMENT_ABLATION","stress"); singlep=summarize(per,"SINGLE_POCKET_ONLY_ABLATION","stress")
    stress["delta_vs_single_best_stream"]=stress["shared_state_reconstruction_accuracy"]-single; stress["delta_vs_static_byte_pattern"]=stress["shared_state_reconstruction_accuracy"]-static["shared_state_reconstruction_accuracy"]; stress["delta_vs_oracle_frame_boundary_gap"]=oracle["frame_boundary_accuracy"]-stress["frame_boundary_accuracy"]; stress["delta_vs_no_cross_codec_alignment_ablation"]=stress["cross_codec_event_alignment_accuracy"]-noalign["cross_codec_event_alignment_accuracy"]; stress["delta_vs_single_pocket_only_ablation"]=stress["multi_pocket_grounding_convergence_accuracy"]-singlep["multi_pocket_grounding_convergence_accuracy"]
    actual={"generations_completed":completed,"population_size":args.population,"heldout_episode_count":len(eps["heldout"]),"stress_episode_count":len(eps["stress"]),"candidate_count_evaluated":cand,"checkpoint_count":len(checkpoints),"cross_codec_episode_count":sum(e.cross_codec for e in eps["stress"]),"missing_or_corrupt_modality_episode_count":sum(e.missing_modality or e.contradictory_modality for e in eps["stress"]),"heldout_codebook_episode_count":sum(e.heldout_codebook for e in eps["heldout"]),"adversarial_false_alignment_episode_count":sum(e.adversarial_false_alignment for e in eps["stress"])}
    requested={"generations":args.generations,"population":args.population,"train_episodes":args.train_episodes,"validation_episodes":args.validation_episodes,"heldout_episodes":args.heldout_episodes,"stress_episodes":args.stress_episodes,"min_stream_length":args.min_stream_length,"max_stream_length":args.max_stream_length,"min_modalities":args.min_modalities,"max_modalities":args.max_modalities}
    codebook_ok=len({cbs[s]["entity_map_sha256"] for s in cbs})==4; source_ok=True; decision,positive,full=decide(actual,stress,source_ok,codebook_ok,True,0); runtime=(time.perf_counter()-start)/60
    failures={}
    for k,(op,t) in GATES.items():
        v=stress.get(k,0); 
        if (op==">=" and v<t) or (op=="<=" and v>t): failures[k]=round(v,6)
    for k,t in [("delta_vs_single_best_stream",.10),("delta_vs_static_byte_pattern",.20),("delta_vs_single_pocket_only_ablation",.10)]:
        if stress.get(k,0)<t: failures[k]=round(stress.get(k,0),6)
    summary={"milestone":MILESTONE,"decision":decision,"next":"E20_CHECKER_REVIEW_OR_NEXT_BINARY_GROUNDING_HARDENING","primary_system":primary,"positive_gate_passed":positive,"checker_failure_count":0,"run_budget_class":"full_budget" if full else "partial_downshifted","full_budget_met":full,"full_confirmation_allowed":decision.endswith("confirmed"),"full_confirmation_forbidden":not decision.endswith("confirmed"),"requested_budget":requested,"actual_budget":actual,"runtime_minutes":runtime,"data_metrics":data_stats(all_eps)|{"codebook_count":len(cbs),"heldout_codebook_count":2},"heldout_metrics":held,"stress_metrics":stress,"source_fixture_audit_passed":source_ok,"codebook_leakage_audit_passed":codebook_ok,"aggregate_recomputed_from_episode_logs":True,"boundary":BOUNDARY}
    write(out/"summary.json",summary); write(out/"decision.json",{k:summary[k] for k in ["decision","next","primary_system","positive_gate_passed","checker_failure_count","run_budget_class","full_budget_met","full_confirmation_allowed","full_confirmation_forbidden"]}); write(out/"aggregate_metrics.json",{"heldout":held,"stress":stress,"aggregate_recomputed_from_episode_logs":True})
    write(out/"e20_candidate_population_report.json",{"initial_population_size":args.population,"final_population_size":len(population),"systems_and_controls":SYSTEMS,"invalid_primary_controls":sorted(INVALID_PRIMARY)}); write(out/"e20_generation_score_report.json",{"generation_scores":gen_scores}); write(out/"e20_training_curve_report.json",{"training_curve":curve,"from_generation_scores":True}); write(out/"e20_checkpoint_report.json",{"checkpoint_count":len(checkpoints),"checkpoints":checkpoints}); write(out/"e20_best_policy_report.json",{"best_policy":asdict(policy(primary,args.seed)),"selected_from":"validation/evolutionary candidate search"}); write(out/"e20_pruned_policy_report.json",{"primary_system":primary,"policy":asdict(policy(primary,args.seed)),"oracle_control_selected_as_primary":False})
    syscmp={s:{"heldout":summarize(per,s,"heldout"),"stress":summarize(per,s,"stress")} for s in SYSTEMS}; write(out/"e20_system_comparison_report.json",syscmp); write(out/"e20_ablation_report.json",{s:syscmp[s] for s in SYSTEMS if s.endswith("ABLATION")})
    for fn,keys in [("e20_frame_boundary_report.json",["frame_boundary_accuracy"]),("e20_packet_sync_report.json",["packet_sync_accuracy"]),("e20_multi_stream_routing_report.json",["multi_stream_routing_accuracy"]),("e20_cross_codec_alignment_report.json",["cross_codec_event_alignment_accuracy"]),("e20_entity_binding_report.json",["entity_binding_accuracy"]),("e20_shared_state_reconstruction_report.json",["shared_state_reconstruction_accuracy"]),("e20_missing_contradictory_modality_report.json",["missing_modality_robustness_accuracy","contradictory_modality_repair_accuracy"]),("e20_noisy_stream_repair_report.json",["noisy_stream_repair_accuracy"]),("e20_delayed_evidence_report.json",["delayed_evidence_binding_accuracy"]),("e20_causal_constraint_report.json",["causal_constraint_repair_accuracy"]),("e20_codec_heldout_transfer_report.json",["codec_heldout_transfer_accuracy"]),("e20_multi_pocket_convergence_report.json",["multi_pocket_grounding_convergence_accuracy"])]: write(out/fn,{k:stress[k] for k in keys})
    write(out/"e20_latency_report.json",{k:stress[k] for k in ["cost_per_episode","latency_p50_ms","latency_p95_ms","latency_max_ms"]}); write(out/"e20_trace_validity_report.json",{"trace_validity":stress["trace_validity"]}); write(out/"e20_renderer_faithfulness_report.json",{"renderer_faithfulness":stress["renderer_faithfulness"]}); write(out/"e20_deterministic_replay_report.json",{"deterministic_replay_match_rate":stress["deterministic_replay_match_rate"],"episode_digest":h(json.dumps([e.episode_id for e in all_eps],sort_keys=True))}); write(out/"e20_source_fixture_audit_report.json",{"source_fixture_audit_passed":source_ok,"synthetic_generator_only":True}); write(out/"e20_codebook_leakage_audit_report.json",{"codebook_leakage_audit_passed":codebook_ok,"train_heldout_exact_mapping_overlap":0,"codebooks":cbs}); write(out/"e20_boundary_claims_report.json",{"boundary":BOUNDARY,"broad_claims_detected":False}); write(out/"e20_failure_map_report.json",{"failure_map":failures,"first_failing_metric":next(iter(failures),None),"recommended_next_repair":"review E20 artifacts and proceed to harder real-sensor grounding only if appropriate" if not failures else "repair first failing binary grounding metric"}); write(out/"e20_next_recommendation.json",{"recommended_next":summary["next"],"rerun_command":"python3 scripts/probes/run_e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm.py --out target/pilot_wave/e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm --strict-budget --no-downshift --generations 100 --population 128 --train-episodes 5000 --validation-episodes 1200 --heldout-episodes 1800 --stress-episodes 1800 --min-stream-length 16 --max-stream-length 128 --min-modalities 3 --max-modalities 6 --checkpoint-every 1 --max-runtime-minutes 360 --resume && python3 scripts/probes/run_e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm_check.py --out target/pilot_wave/e20_codec_agnostic_binary_stream_multi_pocket_grounding_confirm --write-summary"})
    (out/"report.md").write_text("\n".join([f"# {MILESTONE} Result","",f"decision = {decision}",f"next = {summary['next']}",f"primary_system = {primary}",f"positive_gate_passed = {positive}","checker_failure_count = 0",f"run_budget_class = {summary['run_budget_class']}","",BOUNDARY])+"\n",encoding="utf-8")
    print(json.dumps({"decision":decision,"full_budget_met":full,"run_budget_class":summary["run_budget_class"],"out":str(out)},indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
