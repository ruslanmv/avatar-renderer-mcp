#!/usr/bin/env python3
"""
scripts/benchmark.py

Benchmark the Avatar Renderer Pod performance:
- Measures /avatars list latency
- Measures end-to-end render job latency (without GPU work)
- Calculates average response times over N iterations
"""

import argparse
import time
import uuid
import requests
import json
import sys

# Default endpoints
DEFAULT_HOST = "http://localhost:8080"
LIST_ENDPOINT = "/avatars"
RENDER_ENDPOINT = "/render"
STATUS_ENDPOINT = "/status"

def time_request(method, url, **kwargs):
    """Send a request and return (response, elapsed_time)."""
    start = time.time()
    resp = requests.request(method, url, **kwargs)
    elapsed = time.time() - start
    return resp, elapsed

def benchmark_list(host, iterations):
    url = f"{host}{LIST_ENDPOINT}"
    latencies = []
    print(f"Benchmarking GET {LIST_ENDPOINT} for {iterations} iterations...")
    for i in range(iterations):
        resp, t = time_request("GET", url)
        if resp.status_code != 200:
            print(f"  ❌ Iteration {i+1}: status {resp.status_code}")
        else:
            latencies.append(t)
            print(f"  ✅ Iteration {i+1}: {t*1000:.1f} ms")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nAverage list latency: {avg*1000:.1f} ms")
    else:
        print("\nNo successful iterations for list endpoint.")

def benchmark_render(host, avatar_id, audio_url, iterations):
    render_url = f"{host}{RENDER_ENDPOINT}"
    status_url_base = f"{host}{STATUS_ENDPOINT}"
    latencies = []
    print(f"\nBenchmarking POST {RENDER_ENDPOINT} (no GPU work) for {iterations} iterations...")
    for i in range(iterations):
        job_id = str(uuid.uuid4())
        payload = {"avatarId": avatar_id, "voiceUrl": audio_url}
        # Submit render job
        resp, t_submit = time_request("POST", render_url, json=payload)
        if resp.status_code != 200:
            print(f"  ❌ Iteration {i+1}: submit status {resp.status_code}")
            continue
        j = resp.json()
        sid = j.get("statusUrl", "").lstrip("/")
        # Poll immediately once: no actual GPU work so status should be processing or ready fast
        resp2, t_status = time_request("GET", f"{host}/{sid}")
        total_t = t_submit + t_status
        latencies.append(total_t)
        print(f"  ✅ Iteration {i+1}: submit {t_submit*1000:.1f} ms + status {t_status*1000:.1f} ms = total {total_t*1000:.1f} ms")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nAverage render round-trip latency: {avg*1000:.1f} ms")
    else:
        print("\nNo successful iterations for render endpoint.")

def main():
    parser = argparse.ArgumentParser(description="Avatar Service Benchmark Tool")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Base URL of the avatar service")
    parser.add_argument("--avatar-id", default="alice", help="Avatar ID to test against")
    parser.add_argument("--audio-url", default="https://example.com/sample.wav", help="Public URL of a sample WAV file")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Number of iterations per test")
    args = parser.parse_args()

    try:
        # List avatars
        benchmark_list(args.host, args.iterations)
        # Render benchmark (assumes short-circuit/no GPU work)
        benchmark_render(args.host, args.avatar_id, args.audio_url, args.iterations)
    except requests.ConnectionError as e:
        print(f"❌ Connection error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
