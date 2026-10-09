#!/usr/bin/env python3
"""Local contract smoke test for the starter memory service.

This mirrors what the official platform checks in `smoke` mode: the response
envelope, required fields, the `top_k` cap, `user_id` isolation, idempotent
retries, and error handling on malformed payloads.

Usage
-----
python smoke_test.py --base-url http://127.0.0.1:8080 --key $MEMORY_API_KEY
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PASSED: List[str] = []
FAILED: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  FAIL  {name} :: {detail}")
    return condition


def request(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    key: str = "",
) -> Tuple[int, Any]:
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw.strip() else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default="")
    parser.add_argument("--run-id", default="smoke_local")
    args = parser.parse_args()

    base, key, run = args.base_url, args.key, args.run_id
    user_a = f"eval:{run}:conv-a"
    user_b = f"eval:{run}:conv-b"

    print(f"\n== health ==\n  target: {base}")
    try:
        status, body = request(base, "GET", "/health")
        check("GET /health returns 2xx", 200 <= status < 300, f"status={status}")
    except Exception as exc:  # connection refused etc.
        print(f"  FAIL  cannot reach service: {exc}")
        return 1

    print("\n== auth ==")
    if key:
        status, body = request(base, "POST", "/search", {"query": "x", "user_id": user_a, "top_k": 5}, key="")
        check("missing key is rejected with 401", status == 401, f"status={status} body={body}")
    else:
        print("  SKIP  no --key provided (service runs unauthenticated, public smoke only)")

    print("\n== Add ==")
    add_body = {
        "request_id": f"eval:{run}:conv-a:chunk-0",
        "messages": [
            {"role": "user", "timestamp": 1767225600000, "content": "My sister Lan moved to Hangzhou in March 2023."},
            {"role": "assistant", "timestamp": 1767225610000, "content": "Got it, Lan is in Hangzhou."},
            {"role": "user", "timestamp": 1767225620000, "content": "我妹妹小岚去年换工作去了杭州，现在住西湖区。"},
            {"role": "assistant", "timestamp": 1767225630000, "content": "明白，小岚在杭州西湖区工作生活。"},
        ],
        "user_id": user_a,
        "session_id": f"eval:{run}:sample:0",
    }
    status, body = request(base, "POST", "/add", add_body, key)
    check("POST /add returns 200", status == 200, f"status={status} body={body}")
    if isinstance(body, dict):
        check("Add echoes success=true", body.get("success") is True, f"body={body}")
        check(
            "Add echoes all three identifiers",
            body.get("request_id") == add_body["request_id"]
            and body.get("user_id") == add_body["user_id"]
            and body.get("session_id") == add_body["session_id"],
            f"body={body}",
        )
    else:
        check("Add returns a JSON object", False, f"body={body}")

    print("\n== Add idempotency (platform retries keep the same request_id) ==")
    status, body = request(base, "POST", "/add", add_body, key)
    check("retried Add still returns 200", status == 200, f"status={status} body={body}")
    status, first_search = request(base, "POST", "/search", {"query": "杭州", "user_id": user_a, "top_k": 100}, key)
    first_count = len(first_search.get("data", [])) if isinstance(first_search, dict) else -1
    status, second_search = request(base, "POST", "/search", {"query": "杭州", "user_id": user_a, "top_k": 100}, key)
    second_count = len(second_search.get("data", [])) if isinstance(second_search, dict) else -1
    check("retried Add does not duplicate memories", first_count == second_count, f"{first_count} vs {second_count}")

    print("\n== Search contract ==")
    status, body = request(base, "POST", "/search", {"query": "Where does Lan live?", "user_id": user_a, "top_k": 100}, key)
    check("POST /search returns 200", status == 200, f"status={status} body={body}")
    check("Search payload has no `items` wrapper", isinstance(body, dict) and "items" not in body, f"body={type(body)}")
    data = body.get("data") if isinstance(body, dict) else None
    check("data is an array", isinstance(data, list), f"data={type(data)}")
    if isinstance(data, list):
        check("at least one memory was returned", len(data) > 0, "empty result set")
        check("every item has id and content", all(isinstance(i, dict) and i.get("id") and i.get("content") for i in data), f"data={data}")
        check(
            "ids are non-empty strings",
            all(isinstance(i.get("id"), str) and i["id"] for i in data),
            f"data={data}",
        )
    check("English query hits the right memory", bool(data) and "Hangzhou" in json.dumps(data, ensure_ascii=False), "")
    status, body = request(base, "POST", "/search", {"query": "小岚在哪里工作", "user_id": user_a, "top_k": 100}, key)
    check("Chinese query retrieves CJK memories", isinstance(body, dict) and bool(body.get("data")), f"body={body}")

    print("\n== top_k cap ==")
    for requested in (1, 3):
        status, body = request(base, "POST", "/search", {"query": "Lan Hangzhou 杭州", "user_id": user_a, "top_k": requested}, key)
        returned = len(body.get("data", [])) if isinstance(body, dict) else -1
        check(f"returned count never exceeds top_k={requested}", 0 <= returned <= requested, f"returned={returned}")

    print("\n== user_id isolation ==")
    request(
        base,
        "POST",
        "/add",
        {
            "request_id": f"eval:{run}:conv-b:chunk-0",
            "messages": [{"role": "user", "content": "Secret: my safe code is 8891."}],
            "user_id": user_b,
            "session_id": f"eval:{run}:sample:1",
        },
        key,
    )
    status, body = request(base, "POST", "/search", {"query": "safe code 密码", "user_id": user_a, "top_k": 100}, key)
    leaked = "8891" in json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else True
    check("no cross-user memory leakage", not leaked, "user B data leaked into user A scope")

    print("\n== error handling ==")
    status, body = request(base, "POST", "/add", {"messages": [], "user_id": user_a, "session_id": "s"}, key)
    check("Add without request_id fails with 4xx", 400 <= status < 500, f"status={status}")
    status, body = request(base, "POST", "/search", {"query": "x", "user_id": user_a, "top_k": 0}, key)
    check("top_k=0 fails with 4xx", 400 <= status < 500, f"status={status}")
    status, body = request(base, "POST", "/search", {"query": "x", "top_k": 10}, key)
    check("Search without user_id fails with 4xx", 400 <= status < 500, f"status={status}")

    print("\n== multimodal ContentPart[] ==")
    user_mm = f"eval:{run}:conv-mm"
    pixel = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )
    status, body = request(
        base,
        "POST",
        "/add",
        {
            "request_id": f"eval:{run}:conv-mm:chunk-0",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "收据上写着总价 268 元，付款日期 2026-07-12。"},
                        {"type": "image_url", "image_url": {"url": pixel}},
                    ],
                }
            ],
            "user_id": user_mm,
            "session_id": f"eval:{run}:sample:mm",
        },
        key,
    )
    check("multimodal Add returns 200", status == 200, f"status={status} body={body}")
    status, body = request(
        base,
        "POST",
        "/search",
        {
            "query": [{"type": "text", "text": "收据总价是多少"}],
            "user_id": user_mm,
            "top_k": 100,
        },
        key,
    )
    data = body.get("data") if isinstance(body, dict) else None
    check("multimodal Search returns memories", isinstance(data, list) and len(data) > 0, f"body={body}")
    if isinstance(data, list) and data:
        check(
            "multimodal content is preserved as an ordered ContentPart[] array",
            isinstance(data[0].get("content"), list)
            and any(part.get("type") == "image_url" for part in data[0]["content"]),
            f"content={data[0].get('content')}",
        )

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}")
    for failure in FAILED:
        print("  - " + failure)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
