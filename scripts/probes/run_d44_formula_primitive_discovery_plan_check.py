#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--check-only',action='store_true'); a=ap.parse_args()
    out=Path(a.out)
    for f in ['summary.json','decision.json','report.md']:
        if not (out/f).exists(): raise SystemExit(f'missing {f}')
    d=json.loads((out/'decision.json').read_text())
    if d['decision']!='d44_formula_primitive_discovery_plan_ready': raise SystemExit('bad decision')
    print(json.dumps({'status':'ok','decision':d['decision'],'next':d['next']},indent=2))

if __name__=='__main__': main()
