"""Convert Mask Matrix panel (id 40) from table to PNG image."""
import sqlite3, json, sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

AUTOFIT_HTML = '''<div style="width:100%;height:100%;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#181b1f;margin:-8px;padding:0;">
<img src="http://localhost:8088/mask_matrix.png" style="max-width:100%;max-height:100%;object-fit:contain;image-rendering:pixelated;" />
</div>'''

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, data, version FROM dashboard WHERE uid = 'diamond-main-v1'")
    row = cur.fetchone()
    db_id, raw_data, version = row
    dash = json.loads(raw_data)

    for p in dash["panels"]:
        if p.get("id") == 40:
            p["type"] = "text"
            p["options"] = {
                "mode": "html",
                "content": AUTOFIT_HTML,
                "code": {"language": "html", "showLineNumbers": False, "showMiniMap": False}
            }
            p["fieldConfig"] = {"defaults": {}, "overrides": []}
            p["targets"] = []
            p["transparent"] = True
            print(f"  Converted Mask Matrix (id 40) to image panel")

    new_version = version + 1
    dash["version"] = new_version
    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (json.dumps(dash), new_version, db_id))
    conn.commit()
    conn.close()
    print(f"Done: v{version} -> v{new_version}")

if __name__ == "__main__":
    main()
