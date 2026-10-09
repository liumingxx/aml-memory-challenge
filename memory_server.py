#!/usr/bin/env python3
"""Minimal, contract-compliant Add / Search memory service for the
Agent Memory Challenge (Agent Memory Leaderboard, Cycle 2).

Design goals
------------
1. Zero third-party dependencies: pure Python standard library + SQLite, so it
   runs anywhere (including Python 3.9) and is trivial to containerise.
2. Strictly follow the published Add / Search contract: field names, types,
   response envelopes, ``top_k`` cap and ``user_id`` isolation.
3. Be a starting point you can improve: swap the lexical scorer for embeddings,
   add a reranker, add fact extraction -- without touching the HTTP layer.

Endpoints
---------
GET  /health   unauthenticated, any 2xx means the service is up
POST /add      synchronous write; must be durable and searchable on HTTP 200
POST /search   retrieval only; never generate the final answer

Environment variables
---------------------
MEMORY_API_KEY    Secret the platform sends. Unset => no auth (local dev only).
MEMORY_DB         SQLite file path. Default: memory.sqlite3
PORT / HOST       Listen address. HOST defaults to 0.0.0.0, PORT to 8080
MEMORY_PREFIX_TIME  "1" prefixes each returned memory with its ISO date
MAX_BODY_BYTES    Request body cap. Default 64 MiB (contract allows 30 MiB payloads)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

API_KEY = os.environ.get("MEMORY_API_KEY", "").strip()
DB_PATH = os.environ.get("MEMORY_DB", "memory.sqlite3")
PREFIX_TIME = os.environ.get("MEMORY_PREFIX_TIME", "0") == "1"
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024 * 1024)))

_WRITE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    mem_id     TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT,
    ts_ms      INTEGER,
    content    TEXT NOT NULL,
    is_parts   INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def db():
    """One short-lived connection per call: commit on success, always close."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _WRITE_LOCK, db() as conn:
        conn.executescript(SCHEMA)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_ms(ts_ms: Optional[int]) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def fingerprint(user_id: str, role: Optional[str], content_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(user_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((role or "").encode("utf-8"))
    digest.update(b"\x00")
    digest.update(content_text.encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def tokenize(text: str) -> List[str]:
    """Latin word tokens + CJK unigrams and bigrams.

    CJK bigrams are the cheapest way to get usable Chinese retrieval without a
    segmentation dependency. Replace this with jieba or a real analyser later.
    """
    lowered = (text or "").lower()
    tokens = _LATIN_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def parts_to_text(parts: Any) -> str:
    """Flatten multimodal ContentPart[] into searchable text (images excluded)."""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
        return "\n".join(chunks)
    return ""


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return parts_to_text(content)


def human_date(ts_ms: Optional[int]) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d")


def index_text(role: Optional[str], content_text: str, ts_ms: Optional[int]) -> str:
    """What we feed the lexical scorer: content plus temporal hints."""
    pieces = [content_text]
    date = human_date(ts_ms)
    if date:
        day = human_date(ts_ms)
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        pieces.append(f"date {day} {dt.strftime('%B %Y')} {dt.year} {dt.month} {dt.day}")
    if role:
        pieces.append(f"speaker {role}")
    return " ".join(pieces)


# --------------------------------------------------------------------------- #
# Write path (Add)
# --------------------------------------------------------------------------- #

class PayloadError(Exception):
    """Raised for contract violations; mapped to HTTP 400/422."""


def extract_message_content(message: Dict[str, Any]) -> Tuple[str, bool]:
    """Return (text-for-index, is_multimodal_parts)."""
    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            raise PayloadError("messages[].content must be a non-empty string")
        return content, False
    if isinstance(content, list):
        if not content:
            raise PayloadError("messages[].content must not be an empty array")
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("text", "image_url"):
                raise PayloadError("content parts must use type 'text' or 'image_url'")
        return parts_to_text(content), True
    raise PayloadError("messages[].content must be a string or a ContentPart[] array")


def handle_add(body: Dict[str, Any]) -> Dict[str, Any]:
    request_id = body.get("request_id")
    user_id = body.get("user_id")
    session_id = body.get("session_id")
    messages = body.get("messages")

    if not isinstance(request_id, str) or not request_id:
        raise PayloadError("request_id is required and must be a non-empty string")
    if not isinstance(user_id, str) or not user_id:
        raise PayloadError("user_id is required and must be a non-empty string")
    if not isinstance(session_id, str) or not session_id:
        raise PayloadError("session_id is required and must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise PayloadError("messages is required and must be a non-empty array")

    rows: List[Tuple[str, str, str, Optional[str], Optional[int], str, int, str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise PayloadError("each entry of messages must be an object")
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise PayloadError("messages[].role must be 'user' or 'assistant'")
        content_text, is_parts = extract_message_content(message)
        ts_raw = message.get("timestamp")
        ts_ms: Optional[int] = None
        if ts_raw is not None:
            if isinstance(ts_raw, bool) or not isinstance(ts_raw, (int, float)):
                raise PayloadError("messages[].timestamp must be Unix milliseconds")
            ts_ms = int(ts_raw)
        raw_content = message.get("content")
        stored = raw_content if is_parts else content_text
        rows.append(
            (
                "mem_" + hashlib.sha1(
                    (request_id + "\x00" + str(len(rows)) + "\x00" + content_text).encode("utf-8")
                ).hexdigest()[:20],
                user_id,
                session_id,
                role,
                ts_ms,
                json.dumps(stored, ensure_ascii=False) if is_parts else stored,
                1 if is_parts else 0,
                fingerprint(user_id, role, content_text),
                utc_now_iso(),
            )
        )

    with _WRITE_LOCK, db() as conn:
        already = conn.execute(
            "SELECT 1 FROM requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if already:
            # Idempotent: the platform may retry the same logical write.
            return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

        conn.execute(
            "INSERT INTO requests (request_id, user_id, session_id, created_at) VALUES (?, ?, ?, ?)",
            (request_id, user_id, session_id, utc_now_iso()),
        )
        conn.executemany(
            """INSERT OR IGNORE INTO memories
               (mem_id, user_id, session_id, role, ts_ms, content, is_parts, fingerprint, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        # Durability contract: `db()` commits before we leave the block, so by
        # the time HTTP 200 is written the memories are persisted and visible
        # to any subsequent Search request.

    return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}


# --------------------------------------------------------------------------- #
# Read path (Search)
# --------------------------------------------------------------------------- #

class Doc:
    __slots__ = ("mem_id", "text", "raw", "is_parts", "ts_ms", "created_at", "tokens", "length")

    def __init__(self, row: sqlite3.Row):
        self.mem_id = row["mem_id"]
        self.raw = row["content"]
        self.is_parts = bool(row["is_parts"])
        self.ts_ms = row["ts_ms"]
        self.created_at = iso_from_ms(row["ts_ms"]) or row["created_at"]
        if self.is_parts:
            try:
                self.text = parts_to_text(json.loads(self.raw))
            except (ValueError, TypeError):
                self.text = ""
        else:
            self.text = self.raw
        self.tokens = tokenize(index_text(row["role"], self.text, self.ts_ms))
        self.length = len(self.tokens)


def load_docs(user_id: str) -> List[Doc]:
    with db() as conn:
        rows = conn.execute(
            "SELECT mem_id, user_id, session_id, role, ts_ms, content, is_parts, created_at "
            "FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [Doc(row) for row in rows]


def bm25_rank(query: str, docs: Sequence[Doc], limit: int) -> List[Tuple[float, Doc]]:
    """BM25 over the user's own memories, plus a small recency tie-breaker."""
    query_tokens = tokenize(query)
    if not query_tokens or not docs:
        return []

    n_docs = len(docs)
    avg_len = sum(d.length for d in docs) / n_docs or 1.0
    df: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc.tokens):
            df[token] = df.get(token, 0) + 1

    k1, b = 1.2, 0.75
    scored: List[Tuple[float, Doc]] = []
    unique_query = set(query_tokens)
    for doc in docs:
        counts: Dict[str, int] = {}
        for token in doc.tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        for token in unique_query:
            tf = counts.get(token, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc.length / avg_len)
            score += idf * (tf * (k1 + 1.0)) / denom
        if score <= 0.0:
            continue
        # Recency nudge: newer memories win ties, older ones are never dropped.
        if doc.ts_ms:
            age_days = max(0.0, (time.time() * 1000 - doc.ts_ms) / 86_400_000.0)
            score *= 1.0 + 0.05 / (1.0 + age_days / 365.0)
        scored.append((score, doc))

    scored.sort(key=lambda item: (-item[0], item[1].mem_id))
    return scored[:limit]


def render_content(doc: Doc) -> Any:
    content: Any = json.loads(doc.raw) if doc.is_parts else doc.raw
    if not PREFIX_TIME:
        return content
    date = human_date(doc.ts_ms)
    if not date:
        return content
    if isinstance(content, str):
        return f"[{date}] {content}"
    if isinstance(content, list):
        return [{"type": "text", "text": f"[{date}]"}] + content
    return content


def handle_search(body: Dict[str, Any]) -> Dict[str, Any]:
    user_id = body.get("user_id")
    top_k = body.get("top_k")
    query = body.get("query")

    if not isinstance(user_id, str) or not user_id:
        raise PayloadError("user_id is required and must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise PayloadError("top_k is required and must be a positive integer")
    if isinstance(query, str):
        if not query.strip():
            raise PayloadError("query must be a non-empty string")
        query_text = query
    elif isinstance(query, list):
        query_text = parts_to_text(query)
        if not query_text.strip():
            raise PayloadError("query parts must contain non-empty text")
    else:
        raise PayloadError("query must be a string or a ContentPart[] array")

    options = body.get("options")
    if options is not None:
        if not isinstance(options, list):
            raise PayloadError("options must be an array when present")

    docs = load_docs(user_id)
    ranked = bm25_rank(query_text, docs, top_k)

    data: List[Dict[str, Any]] = []
    for score, doc in ranked:
        item: Dict[str, Any] = {
            "id": doc.mem_id,
            "content": render_content(doc),
            "score": round(float(score), 6),
        }
        if doc.created_at:
            item["created_at"] = doc.created_at
        data.append(item)

    # Hard contract cap: never return more than top_k.
    return {"data": data[:top_k]}


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def error_payload(status: int, reason: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {"detail": {"reason": reason}}
    if status == 422:
        body["detail"]["errors"] = [reason]
    return body


def authorized(handler: BaseHTTPRequestHandler) -> bool:
    if not API_KEY:
        return True  # local development / public smoke only
    header = handler.headers.get("Authorization", "")
    if header.startswith("Bearer ") and header[7:].strip() == API_KEY:
        return True
    if header.startswith("Token ") and header[6:].strip() == API_KEY:
        return True
    if handler.headers.get("X-Api-Key", "").strip() == API_KEY:
        return True
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = "AMLMemoryStarter/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------- #
    def _send(self, status: int, payload: Any) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _root_page(self) -> str:
        """Human-friendly landing page: opening the URL in a browser should not 404."""
        host = self.headers.get("Host") or f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8080')}"
        auth_state = "已开启（需要 Authorization: Bearer / Token 或 X-Api-Key）" if API_KEY else "未开启（仅可用于本地调试与公开 smoke）"
        return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AML Memory Starter</title>
<style>
 body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:720px;margin:48px auto;padding:0 20px;color:#1f2430;line-height:1.7}}
 h1{{font-size:22px;margin:0 0 4px}} .ok{{display:inline-block;background:#e6f7ef;color:#0b7a4b;border-radius:999px;padding:2px 12px;font-size:13px;font-weight:600}}
 code{{background:#f2f4f8;padding:2px 6px;border-radius:4px;font-size:13px}}
 pre{{background:#f7f8fa;border:1px solid #e6e8ee;border-radius:8px;padding:12px;overflow:auto;font-size:12.5px}}
 table{{border-collapse:collapse;width:100%;font-size:14px}} td{{border-bottom:1px solid #eef0f4;padding:8px 6px;vertical-align:top}}
 .muted{{color:#68707f;font-size:13px}}
</style></head><body>
<h1>AML Memory Starter <span class="ok">running</span></h1>
<p class="muted">Agent Memory Challenge 第二期 · Add / Search 参考服务</p>
<table>
 <tr><td><b>GET /health</b></td><td>健康检查，无鉴权，返回 2xx 即视为正常 → <a href="/health">/health</a></td></tr>
 <tr><td><b>POST /add</b></td><td>写入记忆（同步，返回 200 前必须落库并可检索）</td></tr>
 <tr><td><b>POST /search</b></td><td>检索记忆，返回 <code>{{"data":[...]}}</code>，条数不得超过 <code>top_k</code></td></tr>
</table>
<p class="muted">监听地址：<code>{host}</code> ｜ 鉴权：{auth_state} ｜ 数据库：<code>{DB_PATH}</code></p>
<p class="muted">用浏览器打开 <code>/health</code> 应该看到 <code>{{"status": "ok"}}</code>；能在浏览器里看到本页，说明服务已经在监听这个端口。</p>
<p class="muted">命令行自测（契约检查 24 项）：</p>
<pre>python smoke_test.py --base-url http://127.0.0.1:8080</pre>
</body></html>"""

    def _read_json(self) -> Dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise PayloadError("Content-Length header is required")
        try:
            length = int(length_header)
        except ValueError:
            raise PayloadError("Content-Length must be an integer")
        if length <= 0:
            raise PayloadError("request body must not be empty")
        if length > MAX_BODY_BYTES:
            raise PayloadError("request body exceeds the configured size limit")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise PayloadError("request body must be valid UTF-8 JSON")
        if not isinstance(parsed, dict):
            raise PayloadError("request body must be a JSON object")
        return parsed

    def log_message(self, fmt: str, *args: Any) -> None:  # keep logs structured
        sys.stderr.write("[%s] %s %s\n" % (utc_now_iso(), self.address_string(), fmt % args))

    # -- routes ------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path.rstrip("/") or "/"
        started = time.time()
        if path == "/health":
            self._send(200, {"status": "ok", "service": "aml-memory-starter"})
        elif path == "/":
            # Not part of the platform contract; exists so a human opening the
            # URL in a browser sees a status page instead of a confusing 404.
            self._send_html(200, self._root_page())
        else:
            self._send(404, error_payload(404, f"no route for GET {path}"))
        status = 200 if path in ("/health", "/") else 404
        sys.stderr.write("[%s] GET %s -> %d in %.3fs\n" % (utc_now_iso(), path, status, time.time() - started))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        started = time.time()
        status = 500
        try:
            if path not in ("/add", "/search"):
                status = 404
                self._send(status, error_payload(404, f"no route for POST {path}"))
                return
            if not authorized(self):
                status = 401
                self._send(status, error_payload(401, "invalid or missing credentials"))
                return
            try:
                body = self._read_json()
                result = handle_add(body) if path == "/add" else handle_search(body)
            except PayloadError as exc:
                status = 422
                self._send(status, error_payload(422, str(exc)))
                return
            status = 200
            self._send(status, result)
        except BrokenPipeError:
            status = 499
        except Exception as exc:  # never leak a stack trace to the platform
            status = 500
            try:
                self._send(status, error_payload(500, f"internal error: {type(exc).__name__}"))
            except Exception:
                pass
        finally:
            sys.stderr.write("[%s] POST %s -> %d in %.3fs\n" % (utc_now_iso(), path, status, time.time() - started))


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    init_db()
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    sys.stderr.write(
        "[%s] memory service listening on http://%s:%d (auth=%s, db=%s)\n"
        % (utc_now_iso(), host, port, "on" if API_KEY else "OFF", DB_PATH)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
