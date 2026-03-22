"""Tiny HTTP server for live training dashboard.
Serves v4.2 directory on localhost:8080.
Dashboard auto-polls training_live_data.json every 3 seconds.

Usage: python live_server.py
Then open: http://localhost:8080/training_live.html
"""
import http.server
import os

PORT = 8080
DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIR)
handler = http.server.SimpleHTTPRequestHandler
handler.extensions_map['.json'] = 'application/json'

print(f"Serving {DIR} on http://localhost:{PORT}")
print(f"Open: http://localhost:{PORT}/training_live.html")
print("Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), handler) as httpd:
    httpd.serve_forever()
