"""CLI integration: drive the real sft-harness binary against a fake mlx_lm.server.

Offline (no weights, no network beyond loopback). Exercises the full loop:
HTTP request body -> response parse -> allowlisted tool dispatch (todo_write,
in-process redb) -> tool message -> final answer. Falls back to `cargo run`
so the binary is built if needed.
"""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1] / "sft_harness"


def _fake_server(script: list[str]) -> tuple[HTTPServer, str]:
    state = {"i": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            idx = min(state["i"], len(script) - 1)
            state["i"] += 1
            body = json.dumps({"choices": [{"message": {"content": script[idx]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _run_cli(model_url: str, prompt: str) -> tuple[str, int]:
    r = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--bin",
            "sft-harness-cli",
            "--",
            "--prompt",
            prompt,
            "--model-url",
            model_url,
            "--model",
            "raw",
        ],
        cwd=HARNESS_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    result = json.loads(lines[-1])
    return result["answer"], result["steps"]


def test_cli_todo_write_then_final() -> None:
    srv, url = _fake_server(['[todo_write(op=create, content="step one")]', "done"])
    try:
        answer, steps = _run_cli(url, "plan it")
    finally:
        srv.shutdown()
    assert answer == "done"
    # system + user + assistant + tool
    assert steps == 4
