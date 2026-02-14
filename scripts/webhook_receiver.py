#!/usr/bin/env python3
"""
Local webhook receiver: listens for deployment "ready" callbacks.
Usage: python3 scripts/webhook_receiver.py
Then use 'ngrok http 9999' to get a public URL and use it as 'user_webhook_url' in POST /v1/deployments.
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 9999

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body.decode())
            print("\n[WEBHOOK]", json.dumps(data, indent=2))
            if data.get("status") == "ready":
                print("  -> Deployment ready. endpoint_url:", data.get("endpoint_url"))
        except Exception as e:
            print("[WEBHOOK] Raw content:", body[:500], e)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args))

def main():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Listening for webhooks on: http://0.0.0.0:{PORT}")
    print("To expose publicly: ngrok http", PORT)
    server.serve_forever()

if __name__ == "__main__":
    main()
