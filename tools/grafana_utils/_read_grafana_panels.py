"""Read current Grafana dashboard panel layout."""
import sqlite3, json
DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
# List all dashboards first
cur.execute("SELECT id, slug, title FROM dashboard")
for r in cur.fetchall():
    print(f"  Dashboard: id={r[0]} slug={r[1]} title={r[2]}")
# Try diamond-swarm
cur.execute("SELECT data, version FROM dashboard WHERE slug LIKE '%diamond%' OR slug LIKE '%swarm%'")
row = cur.fetchone()
if not row:
    print("No matching dashboard found")
    conn.close()
    exit()
d = json.loads(row[0])
print(f"\nVersion: {row[1]}")
print(f"Panels: {len(d['panels'])}")
for p in d['panels']:
    pid = p.get('id')
    title = p.get('title', '?')
    ptype = p.get('type', '?')
    gp = p.get('gridPos', {})
    ds = None
    if 'targets' in p and p['targets']:
        ds = p['targets'][0].get('datasource', {})
    elif 'datasource' in p:
        ds = p.get('datasource')
    print(f"  id={pid:3d}  type={ptype:16s}  {gp.get('w',0):2d}x{gp.get('h',0):2d} @ ({gp.get('x',0)},{gp.get('y',0)})  ds={ds}  title={title}")
conn.close()
