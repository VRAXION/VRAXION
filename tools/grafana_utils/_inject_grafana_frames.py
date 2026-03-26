"""
Replace Input/Output/LCX Before/LCX After table panels with
iframe panels pointing to live PNG images served on :8088.
"""

import sqlite3
import json
import sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

def make_text_panel(panel_id, title, img_url, grid_pos):
    """Create a text panel with HTML img tag."""
    return {
        "id": panel_id,
        "title": title,
        "type": "text",
        "gridPos": grid_pos,
        "options": {
            "mode": "html",
            "content": (
                f'<div style="text-align:center;background:#181b1f;padding:4px;">'
                f'<img src="{img_url}" style="width:100%;image-rendering:pixelated;" />'
                f'</div>'
            ),
            "code": {"language": "html", "showLineNumbers": False, "showMiniMap": False}
        },
        "fieldConfig": {"defaults": {}, "overrides": []},
        "transparent": True
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

    # Remove old table panels (ids 50, 51, 52, 43) — replace with image panels
    old_ids = {50, 51, 52, 43}
    panels = [p for p in panels if p.get("id") not in old_ids]
    print(f"  Removed {len(old_ids)} old table panels")

    base_url = "http://localhost:8088"

    # Add new image panels
    panels.append(make_text_panel(50, "Input Frame", f"{base_url}/input.png",
                                  {"h": 5, "w": 12, "x": 0, "y": 51}))
    panels.append(make_text_panel(51, "Output Frame", f"{base_url}/output.png",
                                  {"h": 5, "w": 12, "x": 12, "y": 51}))
    panels.append(make_text_panel(52, "LCX Before", f"{base_url}/lcx_before.png",
                                  {"h": 12, "w": 12, "x": 0, "y": 56}))
    panels.append(make_text_panel(43, "LCX After", f"{base_url}/lcx_after.png",
                                  {"h": 12, "w": 12, "x": 12, "y": 56}))
    print("  Added 4 image panels (Input, Output, LCX Before, LCX After)")

    # Shift scratchpad down
    for p in panels:
        if p.get("id") == 42:
            p["gridPos"] = {"h": 10, "w": 24, "x": 0, "y": 68}

    dash["panels"] = panels
    new_version = version + 1
    dash["version"] = new_version

    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (json.dumps(dash), new_version, db_id))
    conn.commit()
    conn.close()

    print(f"\nDone: version {version} -> {new_version}")
    print("Layout:")
    print("  y=51: [Input Frame w=12]  [Output Frame w=12]   h=5  (PNG images)")
    print("  y=56: [LCX Before w=12]   [LCX After w=12]      h=12 (PNG images)")
    print("  y=68: [LCX Scratchpad w=24]                      h=10")

if __name__ == "__main__":
    main()
