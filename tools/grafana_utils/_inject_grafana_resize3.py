"""Fix table panel sizing to eliminate scrollbars (v60 -> v61)."""
import sqlite3, json

DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT id, data, version FROM dashboard WHERE id=3")
row = cur.fetchone()
dash_id, raw, ver = row
d = json.loads(raw)
panels = d.get('panels', [])

# Table panel IDs: 40 (mask 5x8), 50 (input 8x8), 51 (output 8x8), 52 (lcx_before 8x8), 43 (lcx_after 8x8)
TABLE_IDS = {40, 50, 51, 52, 43}

for p in panels:
    pid = p.get('id')
    if pid not in TABLE_IDS:
        continue

    # Use small cell height so 8 rows fit without scrolling
    if 'options' not in p:
        p['options'] = {}
    p['options']['cellHeight'] = 'sm'
    p['options']['showHeader'] = False
    p['options']['footer'] = {"show": False}

    # Disable column filtering and set narrow columns
    if 'fieldConfig' not in p:
        p['fieldConfig'] = {}
    if 'defaults' not in p['fieldConfig']:
        p['fieldConfig']['defaults'] = {}
    if 'custom' not in p['fieldConfig']['defaults']:
        p['fieldConfig']['defaults']['custom'] = {}
    p['fieldConfig']['defaults']['custom']['minWidth'] = 30
    p['fieldConfig']['defaults']['custom']['filterable'] = False
    p['fieldConfig']['defaults']['custom']['inspect'] = False

    title = p.get('title', '')
    print(f"  Fixed panel {pid}: {title}")

# Layout: all full-width stacked, generous height
# y=46: Mask Matrix (w=24, h=8) - 5 beings x 8 bits
# y=54: Input (w=12, h=12) | Output (w=12, h=12) - 8x8
# y=66: LCX Before (w=12, h=12) | LCX After (w=12, h=12) - 8x8
for p in panels:
    pid = p.get('id')
    if pid == 40:  # Mask Matrix
        p['gridPos'] = {"h": 8, "w": 24, "x": 0, "y": 46}
    elif pid == 50:  # Input Frame
        p['gridPos'] = {"h": 12, "w": 12, "x": 0, "y": 54}
    elif pid == 51:  # Output Frame
        p['gridPos'] = {"h": 12, "w": 12, "x": 12, "y": 54}
    elif pid == 52:  # LCX Before
        p['gridPos'] = {"h": 12, "w": 12, "x": 0, "y": 66}
    elif pid == 43:  # LCX After
        p['gridPos'] = {"h": 12, "w": 12, "x": 12, "y": 66}
    elif pid == 42:  # LCX Scratchpad
        p['gridPos']['y'] = 78
    elif pid == 41:  # Ant-Bit Accuracy
        p['gridPos']['y'] = 88

d['panels'] = panels
new_ver = ver + 1
d['version'] = new_ver

cur.execute("UPDATE dashboard SET data=?, version=? WHERE id=?",
            (json.dumps(d), new_ver, dash_id))
conn.commit()
conn.close()
print(f"Done: v{ver} -> v{new_ver}")
