"""Update image panels with auto-fit HTML - no scrollbars."""
import sqlite3, json, sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

AUTOFIT_HTML = '''<div style="width:100%;height:100%;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#181b1f;margin:-8px;padding:0;">
<img src="{url}" style="max-width:100%;max-height:100%;object-fit:contain;image-rendering:pixelated;" />
</div>'''

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, data, version FROM dashboard WHERE uid = 'diamond-main-v1'")
    row = cur.fetchone()
    db_id, raw_data, version = row
    dash = json.loads(raw_data)

    base = "http://localhost:8088"
    url_map = {
        50: f"{base}/input.png",
        51: f"{base}/output.png",
        52: f"{base}/lcx_before.png",
        43: f"{base}/lcx_after.png",
    }

    for p in dash["panels"]:
        pid = p.get("id")
        if pid in url_map:
            p["options"]["content"] = AUTOFIT_HTML.format(url=url_map[pid])
            p["options"]["mode"] = "html"
            print(f"  Updated panel {pid} '{p.get('title')}'")

    new_version = version + 1
    dash["version"] = new_version
    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (json.dumps(dash), new_version, db_id))
    conn.commit()
    conn.close()
    print(f"Done: v{version} -> v{new_version}")

if __name__ == "__main__":
    main()
