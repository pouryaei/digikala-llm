"""Real LangGraph and Langfuse SDK (in-memory export); no network or paid calls."""
from contextlib import contextmanager
import json

import pytest

from bonus.evaluate_qa_graph import FixtureLLM, parity, request
from bonus.graph import Workflow
from bonus.observability import TraceStore
from bonus.qa_demo import DEMO_CASES


def demo(i, allow=True):
    _, _, query, pid, _ = DEMO_CASES[i]
    return request(query, pid, allow=allow)


@pytest.fixture
def graph(tmp_path):
    pytest.importorskip("langgraph")
    return Workflow(engine="langgraph", store=TraceStore(tmp_path / "trace.sqlite"), export=False, llm_factory=FixtureLLM)


def test_core_contract_on_real_cases(graph):
    result = parity()
    assert result["matched"] == result["total"] == 36


@pytest.mark.parametrize("i,node,attempts", [(0, "qa_extractive", 0), (1, "qa_insufficient", 0),
    (2, "qa_extractive", 0), (3, "qa_validate", 1), (4, "qa_resolve", 0)])
def test_real_graph_controls_gateway(graph, i, node, attempts):
    entry = graph.run(demo(i))
    trace = entry["trace"]
    assert trace["llm_policy"] == DEMO_CASES[i][-1]
    assert trace["api_attempts"] == attempts
    nodes = [event["node"] for event in trace["events"]]
    assert node in nodes
    assert ("qa_generate" in nodes) == bool(attempts)
    assert (entry["result"]["answer_mode"] == "grounded") == bool(attempts)
    assert DEMO_CASES[i][2] not in json.dumps(graph.store.recent(), ensure_ascii=False)


def test_bad_quote_falls_back_with_same_answer(graph):
    baseline = graph.run(demo(3, False))["result"]
    graph.llm_factory = lambda: FixtureLLM(invalid=True)
    entry = graph.run(demo(3))
    assert entry["trace"]["model_outcome"] == "model_evidence_validation_failed"
    assert entry["trace"]["api_attempts"] == 2
    assert "qa_fallback" in [event["node"] for event in entry["trace"]["events"]]
    assert entry["result"]["answer_mode"] == "extractive"
    assert entry["result"]["answer"] == baseline["answer"]
    assert entry["result"]["evidence"] == baseline["evidence"]


def test_export_failure_does_not_repeat_request(graph, monkeypatch):
    from bonus.observability import LangfuseBridge

    class BrokenClient:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            raise OSError("exporter down")
            yield
        def flush(self):
            raise OSError("exporter down")

    monkeypatch.setenv("FOROUHAR_LANGFUSE_ENABLED", "0")
    graph.bridge = LangfuseBridge()
    graph.bridge.client = BrokenClient()
    result = graph.run(demo(3), flush_export=True)
    assert result["result"]["answer_mode"] == "grounded"
    assert result["trace"]["api_attempts"] == 1
    assert result["trace"]["langfuse_status"] == "export_failed"


def test_real_sdk_exports_one_safe_trace_with_children(graph, monkeypatch):
    langfuse = pytest.importorskip("langfuse", minversion="5")
    import httpx
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from bonus.observability import LangfuseBridge

    monkeypatch.setenv("FOROUHAR_LANGFUSE_ENABLED", "0")
    for key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy"):
        monkeypatch.delenv(key, raising=False)
    sent = InMemorySpanExporter()
    http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500)), trust_env=False)
    client = langfuse.Langfuse(public_key="pk-lf-test-forouhar", secret_key="sk-lf-test-forouhar",
        base_url="http://127.0.0.1:9", tracer_provider=TracerProvider(sampler=ALWAYS_ON), span_exporter=sent, httpx_client=http)
    graph.bridge = LangfuseBridge()
    graph.bridge.client = client
    graph.bridge.status = "sdk_ready"
    try:
        entry = graph.run(demo(3), flush_export=True)
        trace = entry["trace"]
        spans = list(sent.get_finished_spans())
        root = next(span for span in spans if span.name == "forouhar-workflow")
        children = [span for span in spans if span is not root]
        assert trace["langfuse_status"] == "submitted_unverified"
        assert trace["langfuse_trace_id"] == f"{root.context.trace_id:032x}"
        assert {span.name for span in children} == {event["node"] for event in trace["events"]}
        assert all(span.parent and span.parent.span_id == root.context.span_id and
                   span.context.trace_id == root.context.trace_id for span in children)
        exported = " ".join(str(dict(span.attributes)) for span in spans)
        assert DEMO_CASES[3][2] not in exported
        assert entry["result"]["evidence"][0]["text"] not in exported
        assert "مشاهده: نمونه" not in exported and "sk-lf-test-forouhar" not in exported
        assert "citations_verified" in exported
    finally:
        client.shutdown()
        http.close()
