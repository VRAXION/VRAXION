"""Read Diamond Main dashboard panel layout."""
import sqlite3, json
DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT data, version FROM dashboard WHERE id=3")
row = cur.fetchone()
d = json.loads(row[0])
print(f"Version: {row[1]}")
print(f"Panels: {len(d['panels'])}")
ds_uid = None
for p in d['panels']:
    pid = p.get('id')
    title = p.get('title', '?')
    ptype = p.get('type', '?')
    gp = p.get('gridPos', {})
    ds = None
    if 'targets' in p and p['targets']:
        ds = p['targets'][0].get('datasource', {})
        if ds and ds.get('uid'):
            ds_uid = ds.get('uid')
    print(f"  id={pid:3d}  type={ptype:16s}  {gp.get('w',0):2d}x{gp.get('h',0):2d} @ ({gp.get('x',0)},{gp.get('y',0)})  title={title}")
print(f"\nDatasource UID: {ds_uid}")
# Also print templating
templ = d.get('templating', {}).get('list', [])
for t in templ:
    print(f"  Template: name={t.get('name')} type={t.get('type')} current={t.get('current',{}).get('value')}")
conn.close()
