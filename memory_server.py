#!/usr/bin/env python3
"""Contract-compliant Add / Search memory service for the Agent Memory
Challenge (Agent Memory Leaderboard).

v0.2 - retrieval quality, with NO generative model anywhere
-----------------------------------------------------------
Every optimisation below is deterministic algorithm / text processing. No LLM
is called in Add or Search, so the platform's "Add/Search model" rule cannot
be engaged. The dense (embedding) layer is optional and off by default; an
embedding model is not a generative model, but leave it off to keep the
submission maximally conservative.

Version history
---------------
v0.1  BM25 baseline (submitted with the first evaluation request)
v0.2  this version - algorithmic retrieval quality, no generative model,
      no third-party dependencies

What v0.2 adds over v0.1
------------------------
1. Relative-time normalisation. "yesterday", "last month", "three weeks ago",
   "去年", "上个月" are resolved against the message timestamp and appended to
   the indexed text as absolute dates, so temporal questions can match.
2. Neighbouring-message windows. Each message is also indexed inside a
   sliding window of its neighbours, so facts that require more than one
   message become retrievable as a single self-contained unit.
3. Option-aware retrieval. Multiple-choice options are used as a second
   query ranking (fused with RRF), never as an answer.
4. Reciprocal Rank Fusion over the available rankings.
5. MMR de-duplication so the returned top_k is not filled with near-copies
   of the same window.
6. BM25 + phrase-coverage bonus, tuned field weighting.

HTTP contract is unchanged from v0.1: same endpoints, same payloads, same
top_k cap, same user_id isolation, synchronous Add.

Environment variables
---------------------
MEMORY_API_KEY      Secret the platform sends. Unset => no auth (dev only).
MEMORY_DB           SQLite path. Default: memory.sqlite3
PORT / HOST         Listen address. HOST=0.0.0.0, PORT=8080
MEMORY_PREFIX_TIME  "1" (default in v0.2) prefixes returned memories with ISO date
MAX_BODY_BYTES      Request body cap. Default 64 MiB

WINDOW_SIZE         Messages per sliding window. Default 3 (0 disables windows)
WINDOW_MIN          Minimum messages required to emit a window. Default 2
MMR_LAMBDA          Relevance vs diversity trade-off, 0..1. Default 0.82
MMR_POOL            Candidates fed into MMR. Default 220
RRF_K               RRF constant. Default 60

EMBED_BACKEND       none (default) | auto | st | hash
                      none : pure algorithm, no model at all
                      auto : use sentence-transformers if installed, else none
                      st   : require sentence-transformers
                      hash : deterministic test backend (plumbing tests only)
EMBED_MODEL         Default: BAAI/bge-small-zh-v1.5
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
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

SERVICE_VERSION = "0.2"

API_KEY = os.environ.get("MEMORY_API_KEY", "").strip()
DB_PATH = os.environ.get("MEMORY_DB", "memory.sqlite3")
PREFIX_TIME = os.environ.get("MEMORY_PREFIX_TIME", "1") == "1"
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024 * 1024)))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "3"))
WINDOW_MIN = int(os.environ.get("WINDOW_MIN", "2"))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.82"))
MMR_POOL = int(os.environ.get("MMR_POOL", "220"))
RRF_K = int(os.environ.get("RRF_K", "60"))

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "none").strip().lower()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5").strip()
EMBED_QUERY_PREFIX = os.environ.get(
    "EMBED_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章："
)
DENSE_TOP_N = int(os.environ.get("DENSE_TOP_N", "60"))
SPARSE_TOP_N = int(os.environ.get("SPARSE_TOP_N", "120"))
BACKFILL_LIMIT = int(os.environ.get("BACKFILL_LIMIT", "2000"))

_WRITE_LOCK = threading.Lock()


def _log(message: str) -> None:
    sys.stderr.write("[%s] %s\n" % (utc_now_iso(), message))


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
    mem_id      TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT,
    ts_ms       INTEGER,
    content     TEXT NOT NULL,
    is_parts    INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL,
    created_at  TEXT NOT NULL,
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
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        if "embed_model" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN embed_model TEXT")
        if "kind" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN kind TEXT DEFAULT 'msg'")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_ms(ts_ms: Optional[int]) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def sha1_hex(text: str, size: int = 20) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:size]


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
_WS_RE = re.compile(r"\s+")


def stem_latin(token: str) -> str:
    """Very light English suffix stripping.

    Deterministic text processing (no model, no dictionary): it only makes
    "work" / "works" / "working" / "worked" collapse to the same key so a
    question phrased in one form still matches a memory written in another.
    Chinese/Japanese/Korean characters pass through unchanged.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def tokenize(text: str) -> List[str]:
    """Latin word stems + CJK unigrams and bigrams (no segmentation deps)."""
    lowered = (text or "").lower()
    tokens = [stem_latin(word) for word in _LATIN_RE.findall(lowered)]
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def token_set(text: str, cap: int = 400) -> frozenset:
    """Deduplicated token set used for cheap similarity / coverage calculations."""
    tokens = tokenize(text)
    if len(tokens) <= cap:
        return frozenset(tokens)
    # Keep the rarest-looking half instead of an arbitrary prefix.
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ordered = sorted(set(tokens), key=lambda t: (counts[t], t))
    return frozenset(ordered[:cap])


# ---- relative time normalisation ------------------------------------------ #

_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10,
}

_EN_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7}
_EN_AGO_RE = re.compile(
    r"\b(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_EN_LAST_RE = re.compile(r"\blast\s+(night|week|month|year)\b", re.IGNORECASE)
_EN_THIS_RE = re.compile(r"\bthis\s+(week|month|year)\b", re.IGNORECASE)
_EN_TODAY_RE = re.compile(r"\b(today|tonight|yesterday|tomorrow)\b", re.IGNORECASE)
_EN_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_EN_MONTHS) + r")\.?\s+(\d{4})\b", re.IGNORECASE
)
_EN_YEAR_RE = re.compile(r"\b(?:in|since|during)\s+(\d{4})\b", re.IGNORECASE)

_CN_DAY_RE = re.compile(r"(前天|昨天|昨日|今天|今日|明天|后天)")
_CN_LAST_RE = re.compile(r"(上周|上星期|这周|这一周|本周|上个月|上一个月|这个月|本月|去年|上一年|今年|明年)")
_CN_AGO_RE = re.compile(
    r"(\d{1,3}|[一二两三四五六七八九十]+)\s*(天|日|周|星期|个?月|年)\s*(?:前|以前)"
)


def _shift_months(base: datetime, months: int) -> datetime:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base.day, 28)
    return base.replace(year=year, month=month, day=day)


def _num_from(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def resolve_relative_dates(text: str, base_ms: Optional[int]) -> List[str]:
    """Return absolute dates implied by relative expressions in *text*.

    Pure text processing: no model, no external service. The resolved dates are
    appended to the indexed text so a question that mentions "2025" or
    "March 2023" can match a memory that only said "last year".
    """
    if not text:
        return []
    base = None
    if base_ms:
        try:
            base = datetime.fromtimestamp(base_ms / 1000.0, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            base = None
    if base is None:
        return []

    out: List[str] = []

    def emit(dt: datetime, granularity: str = "day") -> None:
        if granularity == "day":
            out.append(dt.strftime("%Y-%m-%d"))
            out.append("%d %d %d" % (dt.year, dt.month, dt.day))
        elif granularity == "month":
            out.append(dt.strftime("%Y-%m"))
            out.append("%d %d" % (dt.year, dt.month))
        else:
            out.append(str(dt.year))
        out.append(str(dt.year))

    lower = text.lower()

    # English: N units ago
    for match in _EN_AGO_RE.finditer(lower):
        amount = _num_from(match.group(1))
        unit = match.group(2).lower()
        if not amount:
            continue
        if unit in _EN_UNIT_DAYS:
            emit(base - timedelta(days=amount * _EN_UNIT_DAYS[unit]))
        elif unit.startswith("month"):
            emit(_shift_months(base, -amount), "month")
        else:
            emit(base.replace(year=max(1, base.year - amount)), "year")

    for match in _EN_LAST_RE.finditer(lower):
        unit = match.group(1).lower()
        if unit == "night":
            emit(base - timedelta(days=1))
        elif unit == "week":
            emit(base - timedelta(days=7))
        elif unit == "month":
            emit(_shift_months(base, -1), "month")
        else:
            emit(base.replace(year=max(1, base.year - 1)), "year")

    for match in _EN_THIS_RE.finditer(lower):
        unit = match.group(1).lower()
        emit(base if unit == "year" else base, "month" if unit == "month" else "day")

    for match in _EN_TODAY_RE.finditer(lower):
        word = match.group(1).lower()
        delta = {"today": 0, "tonight": 0, "yesterday": -1, "tomorrow": 1}.get(word, 0)
        emit(base + timedelta(days=delta))

    # Absolute mentions: normalise "March 2023" -> "2023-03" so both forms match.
    for match in _EN_MONTH_YEAR_RE.finditer(text):
        month = _EN_MONTHS.get(match.group(1).lower())
        year = int(match.group(2))
        if month:
            out.append("%04d-%02d" % (year, month))
            out.append("%04d %d" % (year, month))
    for match in _EN_YEAR_RE.finditer(text):
        out.append(match.group(1))

    # Chinese relative expressions
    _CN_DAY_DELTA = {"前天": -2, "昨天": -1, "昨日": -1, "今天": 0, "今日": 0, "明天": 1, "后天": 2}
    for match in _CN_DAY_RE.finditer(text):
        emit(base + timedelta(days=_CN_DAY_DELTA.get(match.group(1), 0)))
    for match in _CN_LAST_RE.finditer(text):
        word = match.group(1)
        if word in ("上周", "上星期"):
            emit(base - timedelta(days=7))
        elif word in ("这周", "这一周", "本周"):
            emit(base)
        elif word in ("上个月", "上一个月"):
            emit(_shift_months(base, -1), "month")
        elif word in ("这个月", "本月"):
            emit(base, "month")
        elif word in ("去年", "上一年"):
            emit(base.replace(year=max(1, base.year - 1)), "year")
        elif word == "今年":
            emit(base, "year")
        elif word == "明年":
            emit(base.replace(year=base.year + 1), "year")
    for match in _CN_AGO_RE.finditer(text):
        amount = _num_from(match.group(1))
        unit = match.group(2)
        if not amount:
            continue
        if unit in ("天", "日"):
            emit(base - timedelta(days=amount))
        elif unit in ("周", "星期"):
            emit(base - timedelta(days=7 * amount))
        elif unit.endswith("月"):
            emit(_shift_months(base, -amount), "month")
        else:
            emit(base.replace(year=max(1, base.year - amount)), "year")

    # De-duplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for value in out:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


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


def human_date(ts_ms: Optional[int]) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d")


def index_text(role: Optional[str], content_text: str, ts_ms: Optional[int]) -> str:
    """Everything the lexical scorer sees for one memory row."""
    pieces = [content_text]
    if ts_ms:
        try:
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            pieces.append(
                "date %s %s %d %d %d"
                % (dt.strftime("%Y-%m-%d"), dt.strftime("%B %Y"), dt.year, dt.month, dt.day)
            )
        except (OverflowError, OSError, ValueError):
            pass
    resolved = resolve_relative_dates(content_text, ts_ms)
    if resolved:
        pieces.append("resolved " + " ".join(resolved))
    if role:
        pieces.append("speaker " + role)
    return " ".join(pieces)


# --------------------------------------------------------------------------- #
# Embedder (optional dense layer). Never allowed to break the service.
# --------------------------------------------------------------------------- #

class Embedder:
    def __init__(self) -> None:
        self.backend = "none"
        self.model_name = "none" if EMBED_BACKEND != "hash" else "hash-test"
        self.dim = 512
        self.error = ""
        self._model = None
        self._np = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def start_background_load(self) -> None:
        threading.Thread(target=self._load, name="embedder-load", daemon=True).start()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _load(self) -> None:
        try:
            if EMBED_BACKEND == "none":
                self.backend = "none"
                return
            if EMBED_BACKEND == "hash":
                self._np = __import__("numpy")
                self.backend = "hash"
                self.dim = 512
                _log("embedder: hash test backend active")
                return
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL, device="cpu")
            self._np = np
            self._model = model
            self.dim = int(model.get_sentence_embedding_dimension())
            self.backend = "st"
            self.model_name = EMBED_MODEL
            _log("embedder: ready model=%s dim=%d" % (EMBED_MODEL, self.dim))
        except Exception as exc:  # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.backend = "none"
            self.model_name = "none"
            _log("embedder: unavailable, pure-algorithm mode (%s)" % self.error)
        finally:
            self._ready.set()

    def encode(self, texts: Sequence[str], is_query: bool = False) -> Optional[Any]:
        if not self._ready.is_set() or self.backend == "none" or not texts:
            return None
        np = self._np
        try:
            if self.backend == "hash":
                return np.vstack([self._hash_vector(t) for t in texts]).astype(np.float32)
            batch = list(texts)
            if is_query and EMBED_QUERY_PREFIX:
                batch = [EMBED_QUERY_PREFIX + t for t in batch]
            with self._lock:
                vectors = self._model.encode(
                    batch, batch_size=32, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            self.error = "%s: %s" % (type(exc).__name__, exc)
            _log("embedder: encode failed (%s)" % self.error)
            return None

    def _hash_vector(self, text: str) -> Any:
        np = self._np
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


EMBEDDER = Embedder()


def embedding_blob(vector: Any) -> bytes:
    return vector.astype("float32").tobytes()


# --------------------------------------------------------------------------- #
# Write path (Add)
# --------------------------------------------------------------------------- #

class PayloadError(Exception):
    """Raised for contract violations; mapped to HTTP 422."""


def extract_message_content(message: Dict[str, Any]) -> Tuple[str, bool]:
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


def _window_line(role: Optional[str], date: Optional[str], text: str) -> str:
    prefix = ""
    if date:
        prefix += "[%s] " % date
    if role:
        prefix += "%s: " % role
    return prefix + _WS_RE.sub(" ", text.strip())


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

    prepared: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
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
        prepared.append({
            "mem_id": "mem_" + sha1_hex(request_id + "\x00m\x00" + str(index) + "\x00" + content_text),
            "role": role,
            "ts_ms": ts_ms,
            "stored": json.dumps(raw_content, ensure_ascii=False) if is_parts else content_text,
            "is_parts": 1 if is_parts else 0,
            "fingerprint": fingerprint(user_id, role, content_text),
            "indexed": index_text(role, content_text, ts_ms),
            "text": content_text,
        })

    # Build sliding-window rows from the tail of the previous chunk plus this one,
    # so multi-message facts become a single retrievable unit.
    windows: List[Dict[str, Any]] = []
    if WINDOW_SIZE >= WINDOW_MIN >= 2:
        tail: List[Dict[str, Any]] = []
        try:
            with db() as conn:
                rows = conn.execute(
                    "SELECT role, content, ts_ms FROM memories "
                    "WHERE user_id = ? AND session_id = ? AND COALESCE(kind, 'msg') = 'msg' "
                    "AND is_parts = 0 "
                    "ORDER BY rowid DESC LIMIT ?",
                    (user_id, session_id, WINDOW_SIZE - 1),
                ).fetchall()
            tail = [
                {"role": r["role"], "ts_ms": r["ts_ms"], "text": r["content"]}
                for r in reversed(rows)
            ]
        except Exception as exc:  # noqa: BLE001 - windows are best-effort
            _log("window: could not load tail (%s)" % exc)

        stream = tail + [{"role": p["role"], "ts_ms": p["ts_ms"], "text": p["text"],
                          "is_parts": p["is_parts"]} for p in prepared]
        start = len(tail)
        for end in range(start + WINDOW_MIN - 1, len(stream)):
            lo = max(0, end - WINDOW_SIZE + 1)
            chunk = [m for m in stream[lo:end + 1] if not m.get("is_parts")]
            if len(chunk) < WINDOW_MIN:
                continue
            # Only emit windows that end inside the current chunk.
            if end < start:
                continue
            lines = [_window_line(m.get("role"), human_date(m.get("ts_ms")), m.get("text", "")) for m in chunk]
            window_text = "\n".join(line for line in lines if line)
            if not window_text:
                continue
            first_ts = next((m.get("ts_ms") for m in chunk if m.get("ts_ms")), None)
            windows.append({
                "mem_id": "mem_" + sha1_hex(user_id + "\x00w\x00" + window_text),
                "role": "window",
                "ts_ms": first_ts,
                "stored": window_text,
                "is_parts": 0,
                "fingerprint": fingerprint(user_id, "window", window_text),
                "indexed": index_text("window", window_text, first_ts),
                "text": window_text,
            })

    all_rows = prepared + windows
    vectors = None
    if EMBEDDER.ready and EMBEDDER.backend != "none":
        vectors = EMBEDDER.encode([row["text"] for row in all_rows])

    now = utc_now_iso()
    with _WRITE_LOCK, db() as conn:
        already = conn.execute(
            "SELECT 1 FROM requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if already:
            return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

        conn.execute(
            "INSERT INTO requests (request_id, user_id, session_id, created_at) VALUES (?, ?, ?, ?)",
            (request_id, user_id, session_id, now),
        )
        for position, row in enumerate(all_rows):
            blob = None
            model_tag = None
            if vectors is not None and position < len(vectors):
                blob = embedding_blob(vectors[position])
                model_tag = EMBEDDER.model_name
            kind = "window" if row["role"] == "window" else "msg"
            conn.execute(
                """INSERT OR IGNORE INTO memories
                   (mem_id, user_id, session_id, role, ts_ms, content, is_parts,
                    fingerprint, created_at, embedding, embed_model, kind)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["mem_id"], user_id, session_id, row["role"], row["ts_ms"], row["stored"],
                 row["is_parts"], row["fingerprint"], now, blob, model_tag, kind),
            )
        # db() commits before returning, so memories are searchable on HTTP 200.

    return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}


# --------------------------------------------------------------------------- #
# Read path (Search)
# --------------------------------------------------------------------------- #

class Doc:
    __slots__ = ("mem_id", "text", "raw", "is_parts", "ts_ms", "created_at", "kind",
                 "tokens", "length", "tset", "embedding", "embed_model")

    def __init__(self, row: sqlite3.Row) -> None:
        self.mem_id = row["mem_id"]
        self.raw = row["content"]
        self.is_parts = bool(row["is_parts"])
        self.ts_ms = row["ts_ms"]
        self.created_at = iso_from_ms(row["ts_ms"]) or row["created_at"]
        keys = row.keys()
        self.kind = (row["kind"] if "kind" in keys else None) or "msg"
        if self.is_parts:
            try:
                self.text = parts_to_text(json.loads(self.raw))
            except (ValueError, TypeError):
                self.text = ""
        else:
            self.text = self.raw
        self.tokens = tokenize(index_text(row["role"], self.text, self.ts_ms))
        self.length = len(self.tokens)
        self.tset = token_set(self.text)
        self.embedding = row["embedding"] if "embedding" in keys else None
        self.embed_model = row["embed_model"] if "embed_model" in keys else None


def load_docs(user_id: str) -> List[Doc]:
    with db() as conn:
        rows = conn.execute(
            "SELECT mem_id, user_id, session_id, role, ts_ms, content, is_parts, "
            "       created_at, embedding, embed_model, kind "
            "FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [Doc(row) for row in rows]


def backfill_embeddings(user_id: str, docs: Sequence[Doc]) -> None:
    if not EMBEDDER.ready or EMBEDDER.backend == "none":
        return
    pending = [d for d in docs if not d.embedding or d.embed_model != EMBEDDER.model_name]
    if not pending:
        return
    pending = pending[:BACKFILL_LIMIT]
    vectors = EMBEDDER.encode([d.text for d in pending])
    if vectors is None:
        return
    try:
        with _WRITE_LOCK, db() as conn:
            for doc, vector in zip(pending, vectors):
                conn.execute(
                    "UPDATE memories SET embedding = ?, embed_model = ? WHERE mem_id = ?",
                    (embedding_blob(vector), EMBEDDER.model_name, doc.mem_id),
                )
        for doc, vector in zip(pending, vectors):
            doc.embedding = embedding_blob(vector)
            doc.embed_model = EMBEDDER.model_name
    except Exception as exc:  # noqa: BLE001
        _log("embedder: backfill failed (%s)" % exc)


def bm25_rank(query_text: str, docs: Sequence[Doc], limit: int) -> List[Tuple[float, Doc]]:
    """BM25 + phrase-coverage bonus + slight recency nudge."""
    query_tokens = tokenize(query_text)
    if not query_tokens or not docs or limit <= 0:
        return []

    n_docs = len(docs)
    avg_len = sum(d.length for d in docs) / n_docs or 1.0
    df: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc.tokens):
            df[token] = df.get(token, 0) + 1

    unique_query = set(query_tokens)
    k1, b = 1.2, 0.75
    scored: List[Tuple[float, Doc]] = []
    for doc in docs:
        counts: Dict[str, int] = {}
        for token in doc.tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        matched = 0
        for token in unique_query:
            tf = counts.get(token, 0)
            if not tf:
                continue
            matched += 1
            idf = math.log(1.0 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc.length / avg_len)
            score += idf * (tf * (k1 + 1.0)) / denom
        if score <= 0.0:
            continue
        # Coverage: how much of the query is actually present.
        coverage = matched / len(unique_query)
        score *= 1.0 + 0.35 * coverage * coverage
        # Windows are self-contained evidence; give them a small edge on ties.
        if doc.kind == "window":
            score *= 1.04
        if doc.ts_ms:
            age_days = max(0.0, (time.time() * 1000 - doc.ts_ms) / 86_400_000.0)
            score *= 1.0 + 0.05 / (1.0 + age_days / 365.0)
        scored.append((score, doc))

    scored.sort(key=lambda item: (-item[0], item[1].mem_id))
    return scored[:limit]


def dense_rank(query_text: str, docs: Sequence[Doc], limit: int) -> List[Tuple[float, Doc]]:
    if limit <= 0 or EMBEDDER.backend == "none":
        return []
    usable = [d for d in docs if d.embedding and d.embed_model == EMBEDDER.model_name]
    if not usable:
        return []
    query_vector = EMBEDDER.encode([query_text], is_query=True)
    if query_vector is None:
        return []
    np = EMBEDDER._np
    try:
        matrix = np.vstack([np.frombuffer(d.embedding, dtype=np.float32) for d in usable])
    except ValueError:
        return []
    scores = matrix @ query_vector[0]
    order = np.argsort(-scores)[:limit]
    return [(float(scores[i]), usable[i]) for i in order]


def rrf_fuse(rankings: Sequence[Sequence[Doc]], docs_by_id: Dict[str, Doc],
             weights: Optional[Sequence[float]] = None) -> List[Tuple[float, Doc]]:
    weights = list(weights) if weights else [1.0] * len(rankings)
    fused: Dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.mem_id] = fused.get(doc.mem_id, 0.0) + weight / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(score, docs_by_id[mem_id]) for mem_id, score in ordered if mem_id in docs_by_id]


def mmr_rerank(scored: Sequence[Tuple[float, Doc]], limit: int,
               lam: float = MMR_LAMBDA, window: int = 15) -> List[Tuple[float, Doc]]:
    """Greedy Maximal Marginal Relevance over token-set Jaccard similarity.

    Keeps the ranking relevant while stopping the top_k from filling up with
    near-duplicate windows of the same conversation region.
    """
    if len(scored) <= 2 or limit <= 0:
        return list(scored[:limit])
    scores = [s for s, _ in scored]
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    pool = [(float(s - lo) / rng, d) for s, d in scored]
    out: List[Tuple[float, Doc]] = []
    recent: List[frozenset] = []
    while pool and len(out) < limit:
        best_index, best_value = 0, -1e9
        for index, (norm_score, doc) in enumerate(pool):
            if recent:
                tset = doc.tset
                similarity = 0.0
                for other in recent:
                    union = len(tset | other)
                    if union:
                        value = len(tset & other) / union
                        if value > similarity:
                            similarity = value
            else:
                similarity = 0.0
            value = lam * norm_score - (1.0 - lam) * similarity
            if value > best_value:
                best_value, best_index = value, index
        norm_score, doc = pool.pop(best_index)
        out.append((lo + norm_score * rng, doc))
        recent.append(doc.tset)
        if len(recent) > window:
            recent.pop(0)
    return out


def render_content(doc: Doc) -> Any:
    content: Any = json.loads(doc.raw) if doc.is_parts else doc.raw
    # Window rows already carry a per-line "[date] role:" prefix, so adding the
    # memory date on top of them would duplicate it.
    if not PREFIX_TIME or doc.kind == "window":
        return content
    date = human_date(doc.ts_ms)
    if not date:
        return content
    if isinstance(content, str):
        return "[%s] %s" % (date, content)
    if isinstance(content, list):
        return [{"type": "text", "text": "[%s]" % date}] + content
    return content


def _options_text(options: Any) -> str:
    if not isinstance(options, list):
        return ""
    parts: List[str] = []
    for option in options:
        if isinstance(option, str):
            parts.append(option)
        elif isinstance(option, dict):
            parts.append(str(option.get("text") or ""))
    return " ".join(p for p in parts if p)


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
    if options is not None and not isinstance(options, list):
        raise PayloadError("options must be an array when present")

    docs = load_docs(user_id)
    if not docs:
        return {"data": []}

    backfill_embeddings(user_id, docs)
    docs_by_id = {doc.mem_id: doc for doc in docs}

    rankings: List[List[Doc]] = []
    weights: List[float] = []

    sparse = bm25_rank(query_text, docs, SPARSE_TOP_N)
    if sparse:
        rankings.append([doc for _s, doc in sparse])
        weights.append(1.0)

    option_text = _options_text(options)
    if option_text:
        option_pairs = bm25_rank(query_text + " " + option_text, docs, SPARSE_TOP_N)
        if option_pairs:
            rankings.append([doc for _s, doc in option_pairs])
            weights.append(0.6)

    if EMBEDDER.ready and EMBEDDER.backend != "none":
        dense = dense_rank(query_text, docs, DENSE_TOP_N)
        if dense:
            rankings.append([doc for _s, doc in dense])
            weights.append(1.0)

    if not rankings:
        return {"data": []}

    fused = rrf_fuse(rankings, docs_by_id, weights)
    pool = fused[:max(top_k, MMR_POOL)]
    selected = mmr_rerank(pool, top_k) if len(pool) > 2 else list(pool[:top_k])

    data: List[Dict[str, Any]] = []
    for score, doc in selected[:top_k]:
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
        return True
    header = handler.headers.get("Authorization", "")
    if header.startswith("Bearer ") and header[7:].strip() == API_KEY:
        return True
    if header.startswith("Token ") and header[6:].strip() == API_KEY:
        return True
    if handler.headers.get("X-Api-Key", "").strip() == API_KEY:
        return True
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = "AMLMemoryStarter/%s" % SERVICE_VERSION
    protocol_version = "HTTP/1.1"

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
        host = self.headers.get("Host") or "%s:%s" % (
            os.environ.get("HOST", "0.0.0.0"), os.environ.get("PORT", "8080"))
        auth_state = "已开启" if API_KEY else "未开启（仅本地调试）"
        dense_state = "未启用（纯算法）" if EMBEDDER.backend == "none" else (
            "已启用 · %s · %s" % (EMBEDDER.backend, EMBEDDER.model_name) if EMBEDDER.ready else "加载中…")
        return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AML Memory Starter v%s</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:720px;margin:48px auto;padding:0 20px;color:#1f2430;line-height:1.7}
 h1{font-size:22px;margin:0 0 4px} .ok{display:inline-block;background:#e6f7ef;color:#0b7a4b;border-radius:999px;padding:2px 12px;font-size:13px;font-weight:600}
 code{background:#f2f4f8;padding:2px 6px;border-radius:4px;font-size:13px}
 pre{background:#f7f8fa;border:1px solid #e6e8ee;border-radius:8px;padding:12px;overflow:auto;font-size:12.5px}
 table{border-collapse:collapse;width:100%%;font-size:14px} td{border-bottom:1px solid #eef0f4;padding:8px 6px;vertical-align:top}
 .muted{color:#68707f;font-size:13px}
</style></head><body>
<h1>AML Memory Starter <span class="ok">running v%s</span></h1>
<p class="muted">Agent Memory Challenge · Add / Search 记忆服务（无生成式模型）</p>
<table>
 <tr><td><b>GET /health</b></td><td>健康检查，无鉴权 → <a href="/health">/health</a></td></tr>
 <tr><td><b>POST /add</b></td><td>同步写入；返回 200 时已可检索</td></tr>
 <tr><td><b>POST /search</b></td><td>返回 <code>{"data":[...]}</code>，条数不超过 <code>top_k</code></td></tr>
</table>
<p class="muted">监听 <code>%s</code> ｜ 鉴权 %s ｜ 数据库 <code>%s</code></p>
<p class="muted">检索管线：BM25（含短语覆盖率加权）→ 选项扩展 → RRF 融合 → MMR 去重 ｜ 窗口大小 %d ｜ RRF k=%d</p>
<p class="muted">向量层：%s</p>
<p class="muted">契约自测（24 项）：</p>
<pre>python smoke_test.py --base-url http://127.0.0.1:8080</pre>
</body></html>""" % (SERVICE_VERSION, SERVICE_VERSION, host, auth_state, DB_PATH,
                   WINDOW_SIZE, RRF_K, dense_state)

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

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s %s\n" % (utc_now_iso(), self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        started = time.time()
        if path == "/health":
            self._send(200, {
                "status": "ok",
                "service": "aml-memory-starter",
                "version": SERVICE_VERSION,
                "retrieval": "hybrid" if (EMBEDDER.ready and EMBEDDER.backend != "none") else "algorithmic",
                "generative_model_used": False,
                "embedder": {
                    "backend": EMBEDDER.backend,
                    "model": EMBEDDER.model_name,
                    "dim": EMBEDDER.dim,
                    "ready": EMBEDDER.ready,
                },
            })
        elif path == "/":
            self._send_html(200, self._root_page())
        else:
            self._send(404, error_payload(404, "no route for GET %s" % path))
        status = 200 if path in ("/health", "/") else 404
        sys.stderr.write("[%s] GET %s -> %d in %.3fs\n" % (utc_now_iso(), path, status, time.time() - started))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        started = time.time()
        status = 500
        try:
            if path not in ("/add", "/search"):
                status = 404
                self._send(status, error_payload(404, "no route for POST %s" % path))
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
                self._send(status, error_payload(500, "internal error: %s" % type(exc).__name__))
            except Exception:
                pass
        finally:
            sys.stderr.write("[%s] POST %s -> %d in %.3fs\n" % (utc_now_iso(), path, status, time.time() - started))


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    init_db()
    _log("starting v%s (db=%s, windows=%d, embed=%s)"
         % (SERVICE_VERSION, DB_PATH, WINDOW_SIZE, EMBED_BACKEND))
    if EMBED_BACKEND != "none":
        EMBEDDER.start_background_load()
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
