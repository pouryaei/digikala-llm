"""Local, versioned JSON response cache using the standard library SQLite."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
TTL = 86400


def make_cache_key(model, prompt_version, question, reviews, config=None):
    payload = {"model": model, "prompt_version": prompt_version, "question": question,
               "reviews": reviews, "config": config or {}}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _connection():
    path = Path(os.environ.get("QBC13_LLM_CACHE_PATH", str(ROOT / "cache" / "llm" / "task2.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=3)
    con.execute("CREATE TABLE IF NOT EXISTS llm_responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires REAL NOT NULL)")
    return con


def get_cached_response(key):
    try:
        from contextlib import closing
        with closing(_connection()) as con, con:
            row = con.execute("SELECT payload FROM llm_responses WHERE key=? AND expires>?", (key, time.time())).fetchone()
        return json.loads(row[0]) if row else None
    except (OSError, sqlite3.Error, ValueError):
        return None


def save_response(key, response):
    try:
        from contextlib import closing
        with closing(_connection()) as con, con:
            con.execute("INSERT OR REPLACE INTO llm_responses VALUES (?,?,?)", (key, json.dumps(response, ensure_ascii=False, default=str), time.time()+TTL))
    except (OSError, sqlite3.Error, ValueError):
        pass
