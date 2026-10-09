"""Serve the interactive TE Versatility Scout from the repository root."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TE Versatility Scout web app.")
    parser.add_argument("--port", type=int, default=8000, help="Local TCP port to serve.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = lambda *handler_args, **handler_kwargs: SimpleHTTPRequestHandler(
        *handler_args, directory=root, **handler_kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"TE Versatility Scout: http://127.0.0.1:{args.port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
