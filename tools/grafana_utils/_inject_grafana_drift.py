"""Replace LCX Before with LCX Drift, rename LCX After -> LCX State (v61 -> v62)."""
import sqlite3, json

DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
DS_UID = "efd1xv51o3vuob"
DS = {"type": "influxdb", "uid": DS_UID}

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT id, data, version FROM dashboard WHERE id=3")
row = cur.fetchone()
dash_id, raw, ver = row
d = json.loads(raw)
panels = d.get('panels', [])

SIGNED_THRESHOLDS = [
    {"color": "#00838F", "value": None},
    {"color": "#607D8B", "value": 0},
    {"color": "#C2185B", "value": 0.5},
]

def make_flux_query(frame_type):
    return f'''from(bucket: "diamond")
    |> range(start: -30d)
    |> filter(fn: (r) => r._measurement == "frame_snapshot")
    |> filter(fn: (r) => r.run_id == "${{run_id}}")
    |> filter(fn: (r) => r.frame_type == "{frame_type}")
    |> filter(fn: (r) => r._field == "value")
    |> last()
    |> group()
    |> pivot(rowKey: ["row"], columnKey: ["col"], valueColumn: "_value")
    |> sort(columns: ["row"])'''

hide_cols = ["row", "_start", "_stop", "_time", "_measurement",
             "run_id", "frame_type", "_field", "_result", "table", "result"]

def make_overrides():
    return [{"matcher": {"id": "byName", "options": c},
             "properties": [{"id": "custom.hidden", "value": True}]}
            for c in hide_cols]

for p in panels:
    pid = p.get('id')

    # Replace LCX Before (id=52) -> LCX Drift
    if pid == 52:
        p['title'] = 'LCX Drift'
        p['targets'] = [{
            "datasource": DS,
            "query": make_flux_query("lcx_drift"),
            "refId": "A",
        }]
        p['datasource'] = DS
        p['fieldConfig']['overrides'] = make_overrides()
        print(f"  Converted panel 52: LCX Before -> LCX Drift")

    # Rename LCX After (id=43) -> LCX State, update query
    if pid == 43:
        p['title'] = 'LCX State'
        p['targets'] = [{
            "datasource": DS,
            "query": make_flux_query("lcx_state"),
            "refId": "A",
        }]
        p['datasource'] = DS
        p['fieldConfig']['overrides'] = make_overrides()
        print(f"  Renamed panel 43: LCX After -> LCX State")

d['panels'] = panels
new_ver = ver + 1
d['version'] = new_ver

cur.execute("UPDATE dashboard SET data=?, version=? WHERE id=?",
            (json.dumps(d), new_ver, dash_id))
conn.commit()
conn.close()
print(f"Done: v{ver} -> v{new_ver}")
