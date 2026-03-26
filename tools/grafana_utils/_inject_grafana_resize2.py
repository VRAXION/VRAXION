"""Resize Input/Output frame panels taller."""
import sqlite3, json, sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, data, version FROM dashboard WHERE uid = 'diamond-main-v1'")
    row = cur.fetchone()
    db_id, raw_data, version = row
    dash = json.loads(raw_data)

    for p in dash["panels"]:
        pid = p.get("id")
        if pid == 50:  # Input Frame
            p["gridPos"] = {"h": 10, "w": 24, "x": 0, "y": 51}
            print(f"  Input Frame -> h=10")
        elif pid == 51:  # Output Frame
            p["gridPos"] = {"h": 10, "w": 24, "x": 0, "y": 61}
            print(f"  Output Frame -> h=10")
        elif pid == 52:  # LCX Before - shift down
            p["gridPos"] = {"h": 14, "w": 24, "x": 0, "y": 71}
        elif pid == 43:  # LCX After - shift down
            p["gridPos"] = {"h": 14, "w": 24, "x": 0, "y": 85}
        elif pid == 42:  # Scratchpad - shift down
            p["gridPos"] = {"h": 10, "w": 24, "x": 0, "y": 99}

    new_version = version + 1
    dash["version"] = new_version
    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (json.dumps(dash), new_version, db_id))
    conn.commit()
    conn.close()
    print(f"Done: v{version} -> v{new_version}")

if __name__ == "__main__":
    main()
