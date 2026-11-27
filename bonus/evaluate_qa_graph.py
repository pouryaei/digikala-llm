"""Offline parity and paired routing evaluation; real model strictly opt-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter

from bonus.evaluate_workflow import result_key
from bonus.graph import Workflow
from bonus.qa_demo import DEMO_CASES
from src.config import Config
from src.rag import run_rag_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/bonus/qa_graph_ab.json"


class FixtureLLM:
    """Exercises the real prompt/parser/citation path without a network call."""
    model = "offline-fixture-not-a-model"

    def __init__(self, invalid=False):
        self.invalid = invalid

    def generate(self, system, user):
        # The optional schema-repair suffix follows the first JSON object.
        payload = json.JSONDecoder().raw_decode(user)[0]
        witness = payload["retrieved_reviews"][0]
        text = witness["text"].strip()
        answer = {"observations": [{"point": "نمونهٔ آزمون ساختار، نه داوری کیفیت",
                                    "comment_ids": [-1 if self.invalid else witness["comment_id"]],
                                    "quote": text}]}
        return {"text": json.dumps(answer, ensure_ascii=False), "latency": 0, "api_attempts": 1,
                "prompt_tokens": 0, "completion_tokens": 0, "usage_known": False}


def request(query, pid, *, allow=False):
    return {"query": query, "product_ids": [pid] if pid is not None else [],
            "intent": "qa", "use_cache": False, "allow_llm": allow}


def parity():
    source = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))
    cases = [row for group in ("qa", "edge_qa", "stress_qa") for row in source[group]]
    runner = Workflow(engine="langgraph", export=False)
    # Warm up Python services and compile once; timed pairs alternate order.
    q0 = cases[0]["question"]
    p0 = cases[0]["product_id"]
    run_rag_pipeline(q0, p0, source=Config.CATALOG_PATH, generation_mode="extractive", use_cache=False)
    runner.run(request(q0, p0), record=False)
    timings = {"core": [], "graph": []}
    checked = []
    for i, case in enumerate(cases):
        query, pid = case.get("question") or case["query"], case.get("product_id")
        paths = {
            "core": lambda: run_rag_pipeline(query, pid, source=Config.CATALOG_PATH,
                                             generation_mode="extractive", use_cache=False),
            "graph": lambda: runner.run(request(query, pid), record=False)["result"],
        }
        results, durations = {}, {}
        for name in (("core", "graph") if i % 2 == 0 else ("graph", "core")):
            started = perf_counter()
            results[name] = paths[name]()
            durations[name] = (perf_counter()-started)*1000
            timings[name].append(durations[name])
        a, b = results["core"], results["graph"]
        checked.append({"case_id": case["case_id"], "core_status": a["status"], "graph_status": b["status"],
                        "same": result_key(a) == result_key(b) and a["status"] != "error",
                        "witness_ids": [e["comment_id"] for e in b.get("evidence", [])],
                        "core_ms": round(durations["core"], 3), "graph_ms": round(durations["graph"], 3)})
    return {"engine": "LangGraph", "matched": sum(row["same"] for row in checked), "total": len(checked),
            "core_p50_ms": round(statistics.median(timings["core"]), 3),
            "graph_p50_ms": round(statistics.median(timings["graph"]), 3),
            "timing_note": "warm, alternating paired order, result-cache and trace export disabled; dataset-specific",
            "cases": checked}


def packed(entry, live):
    trace, result = entry["trace"], entry["result"]
    content = {"policy": trace["llm_policy"], "model_outcome": trace["model_outcome"], "status": trace["status"],
               "answer_mode": result.get("answer_mode"), "api_attempts": trace["api_attempts"],
               "api_calls": trace["api_calls"], "usage_known": trace["usage_known"],
               "prompt_tokens": trace["prompt_tokens"], "completion_tokens": trace["completion_tokens"],
               "elapsed_ms": round(trace["elapsed_ms"], 3), "evidence_count": trace["evidence_count"],
               "path": [x["node"] for x in trace["events"]]}
    if live:
        # Written only to the explicit local report for manual assessment.
        content.update(answer=result.get("answer"), evidence=result.get("evidence", []))
    return content


def evaluate(*, live=False):
    if live:
        from dotenv import load_dotenv
        from phase4_llm.llm_client import is_configured_key
        import os
        load_dotenv(ROOT / ".env", override=False)
        if not is_configured_key(os.getenv("QBC13_API_KEY")):
            raise ValueError("Set a real QBC13_API_KEY in the local .env before using --live.")
    p = parity()
    if p["matched"] != p["total"]:
        raise AssertionError("The offline graph changed behavior.")
    factory = FixtureLLM
    if live:
        from phase4_llm.llm_client import LLMClient, LLMUnavailable

        class BudgetedLLM(LLMClient):
            remaining = 8

            def __init__(self):
                super().__init__(max_retries=1, max_tokens=350)

            def generate(self, system, user):
                if BudgetedLLM.remaining < 1:
                    raise LLMUnavailable("Experiment attempt cap reached", attempts=0)
                BudgetedLLM.remaining -= 1
                return super().generate(system, user)

        factory = BudgetedLLM
    runners = {name: Workflow(engine="langgraph", export=False, generation_strategy=name,
                              llm_factory=factory) for name in ("always", "selective")}
    rows = []
    for i, (cid, label, query, pid, expected) in enumerate(DEMO_CASES):
        sides = {}
        for name in (("always", "selective") if i % 2 == 0 else ("selective", "always")):
            entry = runners[name].run(request(query, pid, allow=True), record=False)
            sides[name] = packed(entry, live)
        if sides["selective"]["policy"] != expected:
            raise AssertionError(f"{cid}: unexpected selective policy")
        rows.append({"case_id": cid, "label": label, "expected_selective_policy": expected,
                     **({"question": query} if live else {}), **sides})
    if not live:
        _, _, query, pid, _ = DEMO_CASES[3]
        broken = Workflow(engine="langgraph", export=False, llm_factory=lambda: FixtureLLM(invalid=True))
        fallback = packed(broken.run(request(query, pid, allow=True), record=False), False)
        if fallback["answer_mode"] != "extractive" or "qa_fallback" not in fallback["path"]:
            raise AssertionError("Invalid citation escaped fallback.")
    else:
        fallback = None
    comparison = {f"{name}_model_attempts": sum(row[name]["api_attempts"] for row in rows) for name in runners}
    if live:
        from evaluation.cost_from_usage import calculate
        import os
        ir, ore = os.getenv("FOROUHAR_INPUT_USD_PER_MILLION", ""), os.getenv("FOROUHAR_OUTPUT_USD_PER_MILLION", "")
        for name in runners:
            parts = [row[name] for row in rows]
            comparison[f"{name}_gateway_cost"] = calculate(parts, float(ir) if ir else None, float(ore) if ore else None)
    return {"mode": "live_gateway" if live else "offline_stub", "offline_parity": p, "comparison": comparison,
            "paired_cases": rows, "invalid_citation_simulation": fallback,
            "limitations": "Five targeted cases, not representative. Offline fixture tests execution, not quality, tokens, or money. Live quality needs blind human review."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="Opt in to the paid gateway (at most eight attempts across both variants).")
    p.add_argument("--output", type=Path, help="Explicit report path; required for live output (contains demo answers and evidence).")
    args = p.parse_args()
    if args.live and args.output is None:
        p.error("--live requires --output with a local path distinct from the offline report")
    if args.live and args.output.resolve() == DEFAULT_REPORT.resolve():
        p.error("Keep the offline report; choose a separate --output path.")
    result = evaluate(live=args.live)
    out = args.output or DEFAULT_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "parity": {k:v for k,v in result["offline_parity"].items() if k != "cases"},
                      "comparison": result["comparison"], "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
