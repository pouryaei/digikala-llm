"""Isolate cache per test for both the original DiskCache and the Task2 SQLite version."""
import importlib
import pytest


@pytest.fixture(autouse=True)
def isolated_llm_cache(monkeypatch, tmp_path):
    cache_module = importlib.import_module("src.phase4_llm.cache")
    if hasattr(cache_module, "cache"):
        from diskcache import Cache
        with Cache(str(tmp_path / "llm-cache")) as cache:
            monkeypatch.setattr(cache_module, "cache", cache)
            yield
    else:
        monkeypatch.setenv("QBC13_LLM_CACHE_PATH", str(tmp_path / "llm-cache.sqlite3"))
        yield
