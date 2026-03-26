"""Read Grafana datasources."""
import sqlite3, json
DB = r"C:\Program Files\GrafanaLabs\grafana\data\grafana.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT id, uid, name, type, url FROM data_source")
for r in cur.fetchall():
    print(f"  id={r[0]} uid={r[1]} name={r[2]} type={r[3]} url={r[4]}")
conn.close()
