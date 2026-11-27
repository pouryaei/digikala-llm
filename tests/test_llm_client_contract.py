"""SDK contract checks without any real network/API requests."""
import os
import sys
import types
import unittest
from unittest.mock import patch
from src.phase4_llm.llm_client import LLMClient, LLMUnavailable


class FakeProviderError(Exception): pass
class FakeRateLimit(Exception): pass
class FakeTimeout(Exception): pass
class FakeConnection(Exception): pass


class LLMClientTests(unittest.TestCase):
    def test_missing_key_is_a_clear_configuration_error(self):
        with patch.dict(os.environ, {"QBC13_API_KEY":""}):
            with self.assertRaises(LLMUnavailable): LLMClient()

    def test_sdk_configuration_and_token_measurement_offline(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"summary":"ok"}'))], usage=types.SimpleNamespace(prompt_tokens=12, completion_tokens=8))
        def client(**kwargs):
            self.assertEqual(kwargs["api_key"],"synthetic-test-key")
            self.assertEqual(kwargs["max_retries"],0)
            return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        fake = types.SimpleNamespace(OpenAI=client, RateLimitError=FakeRateLimit, APITimeoutError=FakeTimeout, APIConnectionError=FakeConnection, BadRequestError=FakeProviderError)
        with patch.dict(sys.modules,{"openai":fake}), patch.dict(os.environ,{"QBC13_API_KEY":"synthetic-test-key","QBC13_API_BASE_URL":"https://example.invalid/v1"}):
            provider=LLMClient(model="test-model",max_tokens=220)
            result=provider.generate("system","user")
        self.assertEqual(result["prompt_tokens"],12)
        self.assertEqual(result["completion_tokens"],8)
        self.assertEqual(result["api_attempts"],1)
        self.assertEqual(calls[0]["model"],"test-model")
        self.assertEqual(calls[0]["max_tokens"],220)
        self.assertEqual(calls[0]["response_format"],{"type":"json_object"})
