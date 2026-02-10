#!/usr/bin/env python3
"""
Simple HTTP callback receiver for E2E testing.

Starts a local server at http://127.0.0.1:9909/callback that:
- Receives POST requests with quiz results
- Writes the payload to .sisyphus/evidence/callback-received.json
- Logs all received callbacks

Usage:
    python scripts/callback_receiver.py

The server runs until manually stopped (Ctrl+C).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 9909
EVIDENCE_DIR = Path(".sisyphus/evidence")
CALLBACK_FILE = EVIDENCE_DIR / "callback-received.json"


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures callback payloads."""

    def do_POST(self) -> None:
        """Handle POST requests to /callback."""
        if self.path != "/callback":
            self.send_error(404, f"Path {self.path} not found")
            return

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            # Add metadata
            callback_data = {
                "received_at": datetime.now(timezone.utc).isoformat(),
                "headers": dict(self.headers),
                "payload": payload,
            }

            # Ensure evidence directory exists
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

            # Write to file
            with CALLBACK_FILE.open("w") as f:
                json.dump(callback_data, f, indent=2)

            logger.info(
                "Callback received and saved",
                extra={
                    "file": str(CALLBACK_FILE),
                    "payload_keys": list(payload.keys()),
                },
            )

            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in callback", extra={"error": str(e)})
            self.send_error(400, f"Invalid JSON: {e}")

        except Exception as e:
            logger.error(
                "Failed to process callback", extra={"error": str(e)}, exc_info=True
            )
            self.send_error(500, f"Internal error: {e}")

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default HTTP server logging (we use structured logging)."""
        pass


def run_server() -> None:
    """Start the callback receiver server."""
    server = HTTPServer((HOST, PORT), CallbackHandler)

    logger.info(
        "Callback receiver started",
        extra={
            "url": f"http://{HOST}:{PORT}/callback",
            "output_file": str(CALLBACK_FILE),
        },
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Callback receiver stopped by user")
        server.shutdown()


if __name__ == "__main__":
    run_server()
