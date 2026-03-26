"""Inject LCX Delta panel into Grafana dashboard (v59 -> v60)."""
import sqlite3, json

DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT id, data, version FROM dashboard WHERE slug='diamond-swarm'")
row = cur.fetchone()
if not row:
    print("Dashboard not found!")
    exit(1)

dash_id, raw, ver = row
d = json.loads(raw)
panels = d.get('panels', [])

# Find max panel id
max_id = max(p.get('id', 0) for p in panels)
new_id = max_id + 1

# Current layout from v59:
#   y=46: Mask Matrix (w=24, h=5)
#   y=51: Input Frame (w=12, h=5) + Output Frame (w=12, h=5)
#   y=56: LCX Before (w=12, h=12) + LCX After (w=12, h=12)
#   y=68: LCX Scratchpad (w=24, h=10)

# New layout: squeeze Before+After to w=8 each, add Delta w=8 in between
# y=56: [LCX Before w=8] [LCX Delta w=8] [LCX After w=8]

img_html = lambda name, file: (
    f'<div style="display:flex;align-items:center;justify-content:center;'
    f'width:100%;height:100%;overflow:hidden;margin:-16px 0 0 0;">'
    f'<img src="http://localhost:8088/{file}" '
    f'style="max-width:100%;max-height:100%;object-fit:contain;image-rendering:pixelated;" />'
    f'</div>'
)

# Add LCX Delta panel
delta_panel = {
    "id": new_id,
    "title": "LCX Delta (10x)",
    "type": "text",
    "gridPos": {"h": 12, "w": 8, "x": 8, "y": 56},
    "options": {
        "mode": "html",
        "content": img_html("LCX Delta", "lcx_delta.png"),
        "code": {"language": "html", "showLineNumbers": False, "showMiniMap": False}
    },
    "transparent": True,
}
panels.append(delta_panel)

# Resize LCX Before from w=12 to w=8
for p in panels:
    if p.get('title') == 'LCX Before':
        p['gridPos']['w'] = 8
        p['gridPos']['x'] = 0
        print(f"  Resized LCX Before -> w=8, x=0")
    elif p.get('title') == 'LCX After':
        p['gridPos']['w'] = 8
        p['gridPos']['x'] = 16
        print(f"  Resized LCX After -> w=8, x=16")

d['panels'] = panels
new_ver = ver + 1
d['version'] = new_ver

cur.execute("UPDATE dashboard SET data=?, version=? WHERE id=?",
            (json.dumps(d), new_ver, dash_id))
conn.commit()
conn.close()
print(f"Done: v{ver} -> v{new_ver}")
print(f"  Added LCX Delta (10x) panel id={new_id}")
