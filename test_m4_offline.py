from functools import wraps
from unittest.mock import patch
import json
from src.phase4_llm.service import LLMService


class FakeLLM:
    model = "fake"

    def __init__(self, text):
        self.text = text

    def generate(self, system, user):
        return {"text": self.text, "latency": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0}


reviews = [{"comment_id": 1, "text": "خوب"}, {"comment_id": 2, "text": "بد"}]



def _without_cache(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with patch("src.phase4_llm.service.get_cached_response", return_value=None), patch("src.phase4_llm.service.save_response", return_value=None):
            return function(*args, **kwargs)
    return wrapped

@_without_cache
def test_fake_id_is_removed():
    bad = json.dumps({
        "summary": "x",
        "positive_points": [{"point": "a", "comment_ids": [1, 999]}],
        "negative_points": [{"point": "b", "comment_ids": [888]}],
    })
    r = LLMService(llm=FakeLLM(bad)).answer("سؤال تست ۱", reviews)
    assert r["meta"]["citation_warning"] is True
    assert r["answer"]["positive_points"][0]["comment_ids"] == [1]
    assert r["answer"]["negative_points"] == []


@_without_cache
def test_markdown_fence_is_handled():
    ok = "```json\n" + json.dumps({
        "summary": "x",
        "positive_points": [{"point": "a", "comment_ids": [1]}],
        "negative_points": [],
    }) + "\n```"
    r = LLMService(llm=FakeLLM(ok)).answer("سؤال تست ۲", reviews)
    assert r["meta"]["citation_warning"] is False

if __name__=="__main__":
    tests=[(name,func) for name,func in sorted(globals().items()) if name.startswith("test_") and callable(func)]
    for name,func in tests:
        func()
        print("PASS",name)
    print(f"{len(tests)} offline tests passed (no real API calls).")
