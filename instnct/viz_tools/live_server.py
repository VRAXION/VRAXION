"""Tiny HTTP server for INSTNCT unified visualization.
Serves the instnct/ directory on localhost:8080.
The viz auto-polls training_live_data.json every 3 seconds.

Usage: python live_server.py
Then open: http://localhost:8080/instnct_viz.html
"""
import http.server
import os

PORT = 8080
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

os.chdir(DIR)
handler = http.server.SimpleHTTPRequestHandler
handler.extensions_map['.json'] = 'application/json'

print(f"Serving {DIR} on http://localhost:{PORT}")
print(f"Open: http://localhost:{PORT}/instnct_viz.html")
print("Ctrl+C to stop")

with http.server.HTTPServer(("", PORT), handler) as httpd:
    httpd.serve_forever()
