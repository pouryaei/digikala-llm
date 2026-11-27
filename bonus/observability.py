"""Local typed trace store plus opt-in Langfuse span export.

Never logs raw query/review text or credentials to remote observability.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import threading
from time import monotonic
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"bonus_state"
TRACE_PATH=STATE/"observability.sqlite3"
VERSION="forouhar-bonus-2"


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"),default=str).encode()).hexdigest()


class ResultCache:
    """Bounded thread-safe in-memory cache; not used for private/live requests."""
    def __init__(self,capacity=128,ttl=600):
        self.capacity=capacity;self.ttl=ttl;self.values=OrderedDict();self.lock=threading.RLock()

    def get(self,key):
        with self.lock:
            item=self.values.get(key)
            if item is None:return None
            until,result=item
            if monotonic()>=until:
                del self.values[key];return None
            self.values.move_to_end(key)
            return deepcopy(result)

    def put(self,key,result):
        with self.lock:
            self.values[key]=(monotonic()+self.ttl,deepcopy(result));self.values.move_to_end(key)
            while len(self.values)>self.capacity:self.values.popitem(last=False)

    def clear(self):
        with self.lock:self.values.clear()


CACHE=ResultCache()


def cache_key(request,route):
    # Include all settings that can change a local search/QA result.
    manifest=json.loads((ROOT/"storage/catalog_manifest.json").read_text(encoding="utf8"))
    hybrid=ROOT/"bonus_artifacts/retrieval/manifest.json"
    extra=hybrid.stat().st_mtime_ns if hybrid.is_file() else None
    return digest({"version":VERSION,"catalog":manifest["catalog_sha256"],"hybrid_index_version":extra,
                   "route":route,"query":request["query"],"ids":request.get("product_ids",[]),
                   "filters":request.get("filters",{}),"method":request.get("retrieval_mode","auto"),
                   "price_unit":os.getenv("QBC13_SOURCE_PRICE_UNIT",""),"category":request.get("category")})


class TraceStore:
    def __init__(self,path=TRACE_PATH):
        self.path=Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        conn=sqlite3.connect(self.path,timeout=10)
        conn.execute("""CREATE TABLE IF NOT EXISTS traces(
            trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, route TEXT, engine TEXT,
            status TEXT, elapsed_ms REAL, cached INTEGER, api_calls INTEGER, api_attempts INTEGER,
            prompt_tokens INTEGER, completion_tokens INTEGER, usage_known INTEGER,
            detail_json TEXT NOT NULL)""")
        return conn

    def append(self,row):
        # Only safe scalar metadata and node timing are persisted. Never store
        # the actual query, raw review text, user input, or LLM prompt.
        safe={k:row.get(k) for k in ("trace_id","created_at","route","route_reason","engine","status","elapsed_ms",
                    "cached","api_calls","api_attempts","prompt_tokens","completion_tokens","usage_known",
                    "retrieval_method","llm_policy","evidence_count","uncovered_facets","model_outcome","langfuse_status","langfuse_trace_id","events",
                    "professional","decisions","model_usage","attempt_limit","attempts_remaining","estimated_cost_usd","cache_origin","simulated")}
        with self._connect() as c:
            c.execute("INSERT INTO traces VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (safe["trace_id"],safe["created_at"],safe["route"],safe["engine"],safe["status"],
                       safe["elapsed_ms"],int(bool(safe["cached"])),safe["api_calls"] or 0,safe["api_attempts"] or 0,
                       safe["prompt_tokens"] or 0,safe["completion_tokens"] or 0,int(bool(safe["usage_known"])),
                       json.dumps(safe,ensure_ascii=False)))

    def recent(self,limit=200):
        if not self.path.exists():return []
        with self._connect() as c:
            rows=c.execute("SELECT detail_json FROM traces ORDER BY created_at DESC LIMIT ?",[min(max(int(limit),1),10000)]).fetchall()
        return [json.loads(r[0]) for r in rows]


def stats(rows):
    latency=sorted(float(r.get("elapsed_ms",0)) for r in rows)
    return {"requests":len(rows),"cache_rate":sum(bool(r.get("cached")) for r in rows)/len(rows) if rows else 0,
            "errors":sum(r.get("status")=="error" for r in rows),
            "median_ms":statistics.median(latency) if latency else 0,
            "p95_ms":latency[max(0,int(len(latency)*.95+.999999)-1)] if latency else 0,
            "api_calls":sum(r.get("api_calls") or 0 for r in rows),
            "api_attempts":sum(r.get("api_attempts") or 0 for r in rows),
            "prompt_tokens":sum(r.get("prompt_tokens") or 0 for r in rows),
            "completion_tokens":sum(r.get("completion_tokens") or 0 for r in rows),
            "usage_unknown":sum(not bool(r.get("usage_known",False)) for r in rows)}


@lru_cache(maxsize=4)
def _langfuse_client(base_url, public_key, secret_key, environment):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    # Dedicated provider: never subscribe to unrelated application telemetry.
    provider = TracerProvider(sampler=ALWAYS_ON)
    return Langfuse(base_url=base_url, public_key=public_key, secret_key=secret_key,
                    environment=environment, tracer_provider=provider)


class LangfuseBridge:
    """Optional true SDK tracing; local monitoring is always available."""
    def __init__(self):
        self.client=None;self.status="disabled"
        if os.getenv("FOROUHAR_LANGFUSE_ENABLED","0")!="1":return
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_BASE_URL")):
            self.status="credentials_missing";return
        try:
            url = urlsplit(os.environ["LANGFUSE_BASE_URL"])
            if (not url.hostname or url.username or url.password or
                (url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}))):
                self.status = "invalid_base_url"
                return
            self.client = _langfuse_client(os.environ["LANGFUSE_BASE_URL"],
                os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"],
                os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "forouhar"))
            self.status="sdk_ready"
        except ImportError:self.status="sdk_unavailable"
        except Exception:self.status="sdk_initialization_failed"

    @contextmanager
    def span(self,name,*,safe_input=None,as_type="span",trace_context=None):
        if self.client is None:
            yield None;return
        try:
            context={"trace_context":trace_context} if trace_context else {}
            cm=self.client.start_as_current_observation(as_type=as_type,name=name,input=safe_input or {},**context)
            obs=cm.__enter__()
        except Exception:
            self.status="span_failed"
            yield None;return
        try:
            yield obs
        finally:
            # Observability failures must never repeat the business operation.
            try:cm.__exit__(None,None,None)
            except Exception:self.status="export_failed"

    def update(self,obs,output,*,usage=None,model=None):
        if obs is not None:
            try:
                details={}
                if usage is not None:details["usage_details"]=usage
                if model:details["model"]=model
                obs.update(output=output,**details)
            except Exception:self.status="update_failed"

    def flush(self):
        if self.client is not None:
            try:
                self.client.flush()
                if self.status in {"sdk_ready", "queued_unverified"}:self.status="submitted_unverified"
            except Exception:self.status="export_failed"

    @staticmethod
    def trace_context(obs):
        trace=getattr(obs,"trace_id",None) if obs is not None else None
        span=getattr(obs,"id",None) if obs is not None else None
        return {"trace_id":trace,"parent_span_id":span} if trace and span else None


def timestamp():
    return datetime.now(timezone.utc).isoformat()
