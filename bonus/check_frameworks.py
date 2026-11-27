"""Verify graph parity and optionally confirm safe traces on a Langfuse v4 server."""
from __future__ import annotations

import argparse
import json
import os
from time import monotonic, sleep

from bonus.graph import Workflow
from bonus.evaluate_workflow import result_key
from bonus.qa_demo import DEMO_CASES


def verify_observations(trace_id, nodes, wait_seconds):
    """Poll current Observations API using project credentials, without logging them."""
    import requests
    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    until = monotonic() + wait_seconds
    expected = {"forouhar-workflow", *nodes}
    while True:
        try:
            response = requests.get(f"{base}/api/public/v2/observations", auth=auth,
                params={"traceId": trace_id, "fields": "core,basic", "limit": 100}, timeout=4)
            if response.status_code in {401, 403}:
                return "server_auth_rejected"
            if response.status_code == 404:
                return "v2_endpoint_unavailable"
            if response.ok:
                found = {item.get("name") for item in response.json().get("data", [])}
                if expected <= found:
                    return "server_verified"
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            pass
        if monotonic() >= until:
            return "server_not_verified"
        sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-langgraph", action="store_true")
    p.add_argument("--export-langfuse", action="store_true")
    p.add_argument("--verify-server", action="store_true", help="Require observations to appear in Langfuse v4.")
    p.add_argument("--wait", type=int, default=30, help="Seconds to wait per trace for server verification.")
    args = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    if args.verify_server and not args.export_langfuse:
        p.error("--verify-server requires --export-langfuse")
    if args.export_langfuse and (os.getenv("FOROUHAR_LANGFUSE_ENABLED") != "1" or
            not all(os.getenv(k) for k in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))):
        p.error("Set FOROUHAR_LANGFUSE_ENABLED=1, LANGFUSE_BASE_URL and both Langfuse project keys first")
    if not 0 <= args.wait <= 120:
        p.error("--wait must be between 0 and 120")
    from urllib.parse import urlsplit
    if args.export_langfuse:
        url = urlsplit(os.environ["LANGFUSE_BASE_URL"])
        if (not url.hostname or url.username or url.password or
            (url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}))):
            p.error("Use HTTPS or a loopback Langfuse address.")
    old = Workflow(engine="python", export=False)
    graph = Workflow(engine="langgraph" if args.require_langgraph else "auto", export=args.export_langfuse)
    checks = []
    for cid, _, query, pid, _ in DEMO_CASES:
        req = {"query": query, "product_ids": [pid] if pid is not None else [],
               "intent": "qa", "use_cache": False, "allow_llm": False}
        base = old.run(req, record=False)
        actual = graph.run(req, flush_export=args.export_langfuse)
        trace = actual["trace"]
        passed = result_key(base["result"]) == result_key(actual["result"]) and trace["status"] != "error"
        nodes = [event["node"] for event in trace["events"]]
        receipt = trace["langfuse_status"]
        remote = trace["langfuse_trace_id"]
        if args.verify_server and remote:
            receipt = verify_observations(remote, nodes, args.wait)
        checks.append({"case_id": cid, "engine": trace["engine"], "equal_offline_output": passed,
                       "selected_nodes": nodes, "api_attempts": trace["api_attempts"],
                       "langfuse_status": receipt, "langfuse_trace_id": remote})
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if (not all(row["equal_offline_output"] and row["api_attempts"] == 0 for row in checks)
        or (args.export_langfuse and not all(row["langfuse_trace_id"] for row in checks))
        or (args.verify_server and not all(row["langfuse_status"] == "server_verified" for row in checks))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
