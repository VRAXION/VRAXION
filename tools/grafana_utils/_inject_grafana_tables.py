"""Replace all PNG image panels with native Grafana table panels (v59 -> v60).
Removes: Mask Matrix(40), Input Frame(50), Output Frame(51), LCX Before(52), LCX After(43)
         + any delta panel
Adds:    5 table panels with InfluxDB Flux queries, colored cells, proper sizing.
"""
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

# Remove all text (image) panels in the frame area
remove_ids = set()
for p in panels:
    if p.get('type') == 'text' and p.get('id') in (40, 43, 50, 51, 52):
        remove_ids.add(p['id'])
    # Also remove any delta panel we added
    if p.get('title', '').startswith('LCX Delta'):
        remove_ids.add(p['id'])

panels = [p for p in panels if p.get('id') not in remove_ids]
print(f"Removed {len(remove_ids)} image panels: {remove_ids}")

# Color thresholds for signed [-1, 1]: cyan -> gray -> fuchsia
SIGNED_THRESHOLDS = [
    {"color": "#00838F", "value": None},   # base: cyan
    {"color": "#607D8B", "value": 0},       # gray at 0
    {"color": "#C2185B", "value": 0.5},     # fuchsia at +0.5+
]
# Color thresholds for unsigned [0, 1]: cyan -> fuchsia
UNSIGNED_THRESHOLDS = [
    {"color": "#00838F", "value": None},   # base: cyan (0)
    {"color": "#C2185B", "value": 0.5},     # fuchsia (1)
]

def make_table_overrides(num_cols=8, hide_row=True):
    """Generate field overrides to color cells and hide row column."""
    overrides = []
    if hide_row:
        overrides.append({
            "matcher": {"id": "byName", "options": "row"},
            "properties": [
                {"id": "custom.hidden", "value": True}
            ]
        })
    # Hide metadata columns
    for col in ["_start", "_stop", "_time", "_measurement", "run_id",
                "frame_type", "_field", "_result", "table", "result"]:
        overrides.append({
            "matcher": {"id": "byName", "options": col},
            "properties": [
                {"id": "custom.hidden", "value": True}
            ]
        })
    return overrides

def make_flux_query(measurement, frame_type=None, extra_filters=""):
    """Build Flux query for frame data with row/col pivot."""
    filters = f'|> filter(fn: (r) => r._measurement == "{measurement}")\n'
    filters += f'    |> filter(fn: (r) => r.run_id == "${{run_id}}")\n'
    if frame_type:
        filters += f'    |> filter(fn: (r) => r.frame_type == "{frame_type}")\n'
    filters += extra_filters
    return f'''from(bucket: "diamond")
    |> range(start: -30d)
    {filters}    |> filter(fn: (r) => r._field == "value")
    |> last()
    |> group()
    |> pivot(rowKey: ["row"], columnKey: ["col"], valueColumn: "_value")
    |> sort(columns: ["row"])'''

def make_table_panel(panel_id, title, query, gridPos, thresholds, overrides):
    return {
        "id": panel_id,
        "title": title,
        "type": "table",
        "gridPos": gridPos,
        "datasource": DS,
        "targets": [{
            "datasource": DS,
            "query": query,
            "refId": "A",
        }],
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "align": "center",
                    "cellOptions": {"type": "color-background"},
                    "inspect": False,
                    "filterable": False,
                    "minWidth": 40,
                },
                "thresholds": {
                    "mode": "absolute",
                    "steps": thresholds,
                },
                "color": {"mode": "thresholds"},
                "decimals": 2,
            },
            "overrides": overrides,
        },
        "options": {
            "showHeader": False,
            "cellHeight": "lg",
            "footer": {"show": False},
            "frameIndex": 0,
            "sortBy": [{"displayName": "row", "desc": False}],
        },
        "transparent": True,
    }

# --- Mask Matrix (beings x bits, binary on/off) ---
mask_query = '''from(bucket: "diamond")
    |> range(start: -30d)
    |> filter(fn: (r) => r._measurement == "mask_assignment")
    |> filter(fn: (r) => r.run_id == "${run_id}")
    |> filter(fn: (r) => r._field == "assigned")
    |> last()
    |> group()
    |> pivot(rowKey: ["being_id"], columnKey: ["bit_index"], valueColumn: "_value")
    |> sort(columns: ["being_id"])'''

MASK_THRESHOLDS = [
    {"color": "#283237", "value": None},   # dark: not assigned
    {"color": "#C2185B", "value": 1},       # fuchsia: assigned
]

mask_overrides = make_table_overrides(hide_row=False)
mask_overrides.append({
    "matcher": {"id": "byName", "options": "being_id"},
    "properties": [{"id": "custom.hidden", "value": True}]
})

# --- Build all 5 panels ---
new_panels = []

# Layout:
#   y=46: Mask Matrix (w=24, h=6)
#   y=52: Input (w=12, h=10) | Output (w=12, h=10)
#   y=62: LCX Before (w=12, h=10) | LCX After (w=12, h=10)

# 1. Mask Matrix
new_panels.append(make_table_panel(
    panel_id=40, title="Mask Matrix",
    query=mask_query,
    gridPos={"h": 6, "w": 24, "x": 0, "y": 46},
    thresholds=MASK_THRESHOLDS,
    overrides=mask_overrides,
))

# 2. Input Frame
new_panels.append(make_table_panel(
    panel_id=50, title="Input Frame",
    query=make_flux_query("frame_snapshot", "input"),
    gridPos={"h": 10, "w": 12, "x": 0, "y": 52},
    thresholds=UNSIGNED_THRESHOLDS,
    overrides=make_table_overrides(),
))

# 3. Output Frame
new_panels.append(make_table_panel(
    panel_id=51, title="Output Frame",
    query=make_flux_query("frame_snapshot", "output"),
    gridPos={"h": 10, "w": 12, "x": 12, "y": 52},
    thresholds=SIGNED_THRESHOLDS,
    overrides=make_table_overrides(),
))

# 4. LCX Before
new_panels.append(make_table_panel(
    panel_id=52, title="LCX Before",
    query=make_flux_query("frame_snapshot", "lcx_before"),
    gridPos={"h": 10, "w": 12, "x": 0, "y": 62},
    thresholds=SIGNED_THRESHOLDS,
    overrides=make_table_overrides(),
))

# 5. LCX After
new_panels.append(make_table_panel(
    panel_id=43, title="LCX After",
    query=make_flux_query("frame_snapshot", "lcx_after"),
    gridPos={"h": 10, "w": 12, "x": 12, "y": 62},
    thresholds=SIGNED_THRESHOLDS,
    overrides=make_table_overrides(),
))

# Also push down LCX Scratchpad and Ant-Bit Accuracy
for p in panels:
    if p.get('id') == 42:  # LCX Scratchpad
        p['gridPos']['y'] = 72
    if p.get('id') == 41:  # Ant-Bit Accuracy
        p['gridPos']['y'] = 82

panels.extend(new_panels)
d['panels'] = panels
new_ver = ver + 1
d['version'] = new_ver

cur.execute("UPDATE dashboard SET data=?, version=? WHERE id=?",
            (json.dumps(d), new_ver, dash_id))
conn.commit()
conn.close()
print(f"Done: v{ver} -> v{new_ver}")
print(f"  Added 5 table panels (40,50,51,52,43)")
print(f"  Layout: Mask(y=46), Input+Output(y=52), LCX Before+After(y=62)")
