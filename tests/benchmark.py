"""
Benchmark script for desktop-autocomplete API.
Tests infrastructure: latency, cache warming, compact.

Usage:
    python benchmark.py
    python benchmark.py --url http://209.20.159.6:443
    python benchmark.py --max-tokens 50000
    python benchmark.py --data ../../data/session-2026-01-31T04-24-33
"""

import argparse
import json
import time
from pathlib import Path
import requests


def load_session(path: Path) -> tuple[list[dict], list[dict]]:
    with open(path / "context.json") as f:
        clipboard = json.load(f)
    with open(path / "actions.json") as f:
        actions = json.load(f)
    return clipboard, actions


def api_call(method: str, url: str, **kwargs) -> tuple[dict, float]:
    """Make API call, return (response_json, latency)."""
    start = time.time()
    resp = requests.request(method, url, **kwargs)
    resp.raise_for_status()
    return resp.json(), time.time() - start


def run_benchmark(
    url: str, clipboard: list[dict], actions: list[dict], max_tokens: int
):
    print(f"\nBenchmark: {url}")
    print(
        f"Clipboard: {len(clipboard)} blocks | Actions: {len(actions)} blocks | Max tokens: {max_tokens:,}\n"
    )

    # Set clipboard
    resp, latency = api_call("POST", f"{url}/clipboard", json={"blocks": clipboard})
    session_id = resp["session_id"]
    print(f"Clipboard set: {resp['prompt_tokens']:,} tokens ({latency:.2f}s)\n")

    # Add actions, compact when exceeding max_tokens
    compact_count = 0
    for i in range(0, len(actions), 2):
        action_blocks = actions[i : i + 2]
        resp, latency = api_call(
            "POST",
            f"{url}/action",
            json={"session_id": session_id, "blocks": action_blocks},
        )
        tokens = resp["prompt_tokens"]
        print(f"Action {i//2 + 1}: {tokens:,} tokens ({latency:.2f}s)")

        if tokens > max_tokens:
            compact_count += 1
            print(
                f"\n--- Compact #{compact_count} (exceeded {max_tokens:,} tokens) ---"
            )
            resp, latency = api_call(
                "POST", f"{url}/compact", json={"session_id": session_id}
            )
            print(f"Compacted: {resp['prompt_tokens']:,} tokens ({latency:.2f}s)")
            print(f"Summary:\n{resp['summary']}\n")

    print(f"\nDone. Total compacts: {compact_count}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark desktop-autocomplete API")
    parser.add_argument("--url", default="http://localhost:443")
    parser.add_argument("--data", default="../../data/session-2026-02-02T01-05-49")
    parser.add_argument("--max-tokens", type=int, default=80000)
    args = parser.parse_args()

    data_path = (Path(__file__).parent / args.data).resolve()
    if not data_path.exists():
        print(f"Error: Data not found: {data_path}")
        return 1

    clipboard, actions = load_session(data_path)
    run_benchmark(args.url, clipboard, actions, args.max_tokens)


if __name__ == "__main__":
    exit(main() or 0)
