"""Shared checks for the product-search notebooks; no embedding downloads."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct
import time

from .bm25_index import BM25IndexManager, mapping_ids, mapping_digest, product_key


def find_project_root(start=None):
    start=Path(start or Path.cwd()).resolve()
    for root in (start,*start.parents):
        if (root/"src").is_dir() and (root/"storage"/"id_mapping.json").is_file():return root
    raise FileNotFoundError("Project root with src/ and storage/id_mapping.json was not found. Open the notebook in the project folder.")


def load_mapping(root):
    path=Path(root)/"storage"/"id_mapping.json"
    data=json.loads(path.read_text(encoding="utf-8-sig"));mapping_ids(data)
    return data


def inspect_flat_faiss(path):
    """Inspect only the known IndexFlatIP/L2 header without importing faiss."""
    path=Path(path)
    with path.open("rb") as fh:header=fh.read(16)
    if len(header)!=16:raise ValueError("Truncated FAISS index")
    magic,dimension,count=struct.unpack("<4siq",header)
    if magic not in (b"IxFI",b"IxF2"):
        return {"format":"other","dimension":None,"vector_count":None,"header_verified":False}
    if dimension<=0 or count<0 or path.stat().st_size!=45+dimension*count*4:
        raise ValueError("Invalid/truncated IndexFlat file")
    return {"format":magic.decode(),"dimension":dimension,"vector_count":count,"header_verified":True}


def validate_assets(root, *, index_path=None):
    root=Path(root);mapping=load_mapping(root);ids=mapping_ids(mapping)
    manager=BM25IndexManager(index_path=str(index_path or root/"storage"/"bm25_index.pkl"))
    if not manager.load_index():raise FileNotFoundError("bm25_index.pkl is missing. Run scripts/build_search_engine.py --rebuild-bm25 first.")
    lexical=mapping_ids(manager.id_mapping)
    if lexical!=ids:
        raise ValueError(f"BM25/FAISS corpus mismatch: BM25={len(lexical)}, FAISS={len(ids)}, shared={len(set(lexical)&set(ids))}. Rebuild aligned BM25 before evaluating.")
    path=root/"storage"/"faiss_index.bin"
    flat=inspect_flat_faiss(path) if path.is_file() else {"format":"missing","dimension":None,"vector_count":None,"header_verified":False}
    if flat["header_verified"] and flat["vector_count"]!=len(ids):raise ValueError("FAISS binary vector count disagrees with id_mapping.json")
    return manager,mapping,{"bm25_count":len(lexical),"faiss_mapping_count":len(ids),"same_ids_and_order":True,"mapping_sha256":mapping_digest(mapping),"faiss":flat,"bm25_format":manager.metadata.get("format_version","legacy"),"tokenization_version":manager.metadata.get("tokenization_version"),"bm25_backend":type(manager.bm25).__name__}


def load_query_set(path,mapping):
    payload=json.loads(Path(path).read_text(encoding="utf-8-sig"));queries=payload.get("queries")
    if not isinstance(queries,list) or not queries:raise ValueError("queries.json does not contain queries")
    if payload.get("total_queries")!=len(queries):raise ValueError("total_queries disagrees with actual rows")
    query_ids=[q.get("query_id") for q in queries]
    if any(not qid for qid in query_ids) or len(set(query_ids))!=len(query_ids):raise ValueError("Query IDs must be nonempty and unique")
    ids=set(mapping_ids(mapping));targets=[];no_target=[]
    for q in queries:
        if not isinstance(q.get("query"),str) or not q["query"].strip():raise ValueError("Empty query")
        target=q.get("target_product_id")
        if target is None:no_target.append(q);continue
        target=product_key(target)
        if target not in ids:raise ValueError(f"Query {q['query_id']} points to a product outside the current index: {target}")
        targets.append({**q,"target_product_id":target})
    if not targets:raise ValueError("No known-item target queries available")
    return payload,targets,no_target


def make_queries(mapping,seed=42):
    pool=[r for r in mapping if isinstance(r.get("title_fa"),str) and len(r["title_fa"].strip())>10]
    if len(pool)<20:raise ValueError("At least 20 products with titles are required")
    chosen=random.Random(seed).sample(pool,20);types=["short_keyword"]*7+["semantic"]*7+["review_qa"]*6
    rows=[]
    for i,(r,kind) in enumerate(zip(chosen,types),1):
        title=r["title_fa"].strip();cleaned=re.sub(r"\b(مدل|کد|ظرفیت|بسته|عددی|طرح|مجموعه)\b"," ",title);words=cleaned.split()
        if not words:words=title.split()
        core=" ".join(words[:4 if kind=="short_keyword" else 5])
        query=core if kind=="short_keyword" else f"خرید {core} با کیفیت مناسب" if kind=="semantic" else f"آیا کاربران از کیفیت و کارایی {core} راضی هستند؟"
        rows.append({"query_id":f"Q{i:02d}","query":query,"query_type":kind,"target_product_id":product_key(r["product_id"]),"reference_title":title})
    rows.extend([{"query_id":"Q21","query":"کدام دسته‌بندی بیشترین تعداد نظرات کاربران را دارد؟","query_type":"analytical","target_product_id":None,"reference_title":None},
                 {"query_id":"Q22","query":"محصولاتی که اصلاً در فروشگاه وجود ندارند مثل خودروی تسلا","query_type":"edge_case_out_of_domain","target_product_id":None,"reference_title":None}])
    return {"version":"1.1","method":"Method A - Silver Ground Truth (Known-Item Search)","seed":seed,"mapping_sha256":mapping_digest(mapping),"total_queries":len(rows),"queries":rows}


def evaluate_known_items(search,queries,k=5,mode="bm25"):
    if not queries:raise ValueError("No target queries")
    details=[]
    for q in queries:
        start=time.perf_counter();results=search(q["query"],top_k=k);elapsed=time.perf_counter()-start
        ids=[]
        for item in results:
            pid=product_key(item["product_id"])
            if pid not in ids:ids.append(pid)
        ids=ids[:k];target=product_key(q["target_product_id"])
        rank=ids.index(target)+1 if target in ids else None
        details.append({"query_id":q["query_id"],"query_type":q.get("query_type"),"query":q["query"],"target_product_id":target,"retrieved_ids":ids,"rank":rank,"latency_seconds":round(elapsed,6)})
    n=len(details);latencies=[r["latency_seconds"] for r in details]
    return {"mode":mode,"target_queries":n,"Hit@1":sum(r["rank"]==1 for r in details)/n,f"Hit@{k}":sum(r["rank"] is not None for r in details)/n,
            f"MRR@{k}":sum(1/r["rank"] if r["rank"] else 0 for r in details)/n,"median_latency_seconds":statistics.median(latencies),"queries":details}
