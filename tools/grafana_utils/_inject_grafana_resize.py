"""
Resize LCX panels to full width, stacked vertically. No more scrollbars.
"""

import sqlite3
import json
import sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

# New layout: all full width, stacked
LAYOUT = {
    50: {"h": 5,  "w": 24, "x": 0, "y": 51},   # Input Frame
    51: {"h": 5,  "w": 24, "x": 0, "y": 56},   # Output Frame
    52: {"h": 14, "w": 24, "x": 0, "y": 61},   # LCX Before
    43: {"h": 14, "w": 24, "x": 0, "y": 75},   # LCX After
    42: {"h": 10, "w": 24, "x": 0, "y": 89},   # LCX Scratchpad
}

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id, data, version FROM dashboard WHERE uid = 'diamond-main-v1'")
    row = cur.fetchone()
    if not row:
        print("ERROR: dashboard not found")
        sys.exit(1)

    db_id, raw_data, version = row
    dash = json.loads(raw_data)
    panels = dash.get("panels", [])
    print(f"Current: version={version}, panels={len(panels)}")

    for p in panels:
        pid = p.get("id")
        if pid in LAYOUT:
            old = p.get("gridPos", {})
            p["gridPos"] = LAYOUT[pid]
            print(f"  Panel {pid} '{p.get('title','?')}': w={old.get('w')}->{ LAYOUT[pid]['w']}, h={old.get('h')}->{LAYOUT[pid]['h']}")

    dash["panels"] = panels
    new_version = version + 1
    dash["version"] = new_version

    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (json.dumps(dash), new_version, db_id))
    conn.commit()
    conn.close()

    print(f"\nDone: version {version} -> {new_version}")
    print("Layout:")
    print("  y=51: [Input Frame w=24]      h=5")
    print("  y=56: [Output Frame w=24]     h=5")
    print("  y=61: [LCX Before w=24]       h=14")
    print("  y=75: [LCX After w=24]        h=14")
    print("  y=89: [LCX Scratchpad w=24]   h=10")

if __name__ == "__main__":
    main()
