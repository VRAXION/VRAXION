"""
Inject Grafana template variable change: run_id → constant "current", hidden.
This hardcodes the dashboard to always read the current run without user selection.
"""

import sqlite3
import json
import sys

DB_PATH = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id, data, version FROM dashboard WHERE uid = 'diamond-main-v1'")
    row = cur.fetchone()
    if not row:
        print("ERROR: dashboard diamond-main-v1 not found")
        sys.exit(1)

    db_id, raw_data, version = row
    dash = json.loads(raw_data)
    print(f"Current: version={version}")

    # Update template variable
    templating = dash.get("templating", {"list": []})
    var_list = templating.get("list", [])

    # Find and replace the run_id variable
    found = False
    for i, v in enumerate(var_list):
        if v.get("name") == "run_id":
            var_list[i] = {
                "type": "constant",
                "name": "run_id",
                "label": "Run",
                "description": "Always reads current run (hardcoded)",
                "query": "current",
                "current": {
                    "selected": True,
                    "text": "current",
                    "value": "current"
                },
                "hide": 2,
                "skipUrlSync": False
            }
            found = True
            print("  Updated run_id variable: query -> constant 'current', hidden")

    if not found:
        # Add it if it doesn't exist
        var_list.append({
            "type": "constant",
            "name": "run_id",
            "label": "Run",
            "description": "Always reads current run (hardcoded)",
            "query": "current",
            "current": {
                "selected": True,
                "text": "current",
                "value": "current"
            },
            "hide": 2,
            "skipUrlSync": False
        })
        print("  Added run_id constant variable = 'current', hidden")

    templating["list"] = var_list
    dash["templating"] = templating

    # Write back
    new_version = version + 1
    dash["version"] = new_version
    new_data = json.dumps(dash)

    cur.execute("UPDATE dashboard SET data = ?, version = ? WHERE id = ?",
                (new_data, new_version, db_id))
    conn.commit()
    conn.close()

    print(f"\nDone: version {version} -> {new_version}")
    print("Grafana now hardcoded to run_id='current' (no dropdown)")

if __name__ == "__main__":
    main()
