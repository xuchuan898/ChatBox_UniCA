"""Run offline with python -B -m unittest services.test_model_providers."""
from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from services.model_providers import (
    AlbertChat, AlbertClient, AlbertReranker, build_generator,
    build_query_expander, build_reranker, needs_ollama,
)


class AlbertTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"ALBERT_API_KEY": "test-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.docs = [SimpleNamespace(page_content="first"), SimpleNamespace(page_content="second")]

    def test_key_required(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "ALBERT_API_KEY"):
            AlbertClient({})

    def test_key_file_fallback(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "services.model_providers.Path.read_text", return_value=" file-key\n"
        ) as read:
            client = AlbertClient({"albert": {"api_key_file": "api_albert.txt"}})
            self.assertEqual(client.api_key, "file-key")
            read.assert_called_once_with(encoding="utf-8-sig")

    def test_environment_key_takes_priority_over_file(self):
        with patch("services.model_providers.Path.read_text") as read:
            client = AlbertClient({"albert": {"api_key_file": "api_albert.txt"}})
            self.assertEqual(client.api_key, "test-key")
            read.assert_not_called()

    def test_key_file_errors_do_not_leak_content(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("services.model_providers.Path.read_text", side_effect=OSError("private details")):
                with self.assertRaisesRegex(ValueError, "Cannot read") as ctx:
                    AlbertClient({"albert": {"api_key_file": "api_albert.txt"}})
                self.assertNotIn("private details", str(ctx.exception))
            with patch("services.model_providers.Path.read_text", return_value=" \n"):
                with self.assertRaisesRegex(ValueError, "ALBERT_API_KEY"):
                    AlbertClient({"albert": {"api_key_file": "api_albert.txt"}})

    def test_base_url_normalization(self):
        for base in ("https://example.org", "https://example.org/", "https://example.org/v1/"):
            with self.subTest(base=base):
                client = AlbertClient({"albert": {"base_url": base}})
                self.assertEqual(client.base_url, "https://example.org/v1")
        with patch.dict("os.environ", {"ALBERT_BASE_URL": "https://override.org/v1"}):
            self.assertEqual(AlbertClient({}).base_url, "https://override.org/v1")

    def test_custom_key_env_and_timeout(self):
        with patch.dict("os.environ", {"CUSTOM_KEY": "custom"}):
            client = AlbertClient({"albert": {"api_key_env": "CUSTOM_KEY", "timeout": 30}})
            self.assertEqual(client.api_key, "custom")
            self.assertEqual(client.timeout, 30)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AlbertClient({"albert": {"timeout": value}})

    @patch("services.model_providers.urlopen")
    def test_http_auth_and_json(self, urlopen):
        urlopen.return_value.__enter__.return_value = io.BytesIO(b'{"results": []}')
        self.assertEqual(AlbertClient({}).post("rerank", {"query": "café"}), {"results": []})
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "https://albert.api.etalab.gouv.fr/v1/rerank")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(req.data), {"query": "café"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)

    @patch("services.model_providers.urlopen")
    def test_http_failures_hide_response_details(self, urlopen):
        for status in (401, 403, 404, 429, 503):
            urlopen.side_effect = HTTPError("https://example.org", status, "test-key", {}, io.BytesIO(b'test-key'))
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as ctx:
                AlbertClient({}).post("rerank", {})
            self.assertNotIn("test-key", str(ctx.exception))
        urlopen.side_effect = URLError("test-key")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            AlbertClient({}).post("rerank", {})

    @patch("services.model_providers.urlopen")
    def test_invalid_json(self, urlopen):
        for body in (b'not-json', b'[]'):
            urlopen.return_value.__enter__.return_value = io.BytesIO(body)
            with self.subTest(body=body), self.assertRaises(RuntimeError):
                AlbertClient({}).post("rerank", {})

    def test_chat_payload(self):
        client = Mock()
        client.post.return_value = {"choices": [{"message": {"content": "answer"}}]}
        self.assertEqual(AlbertChat(client, "model", 0.2, 77).invoke("question").content, "answer")
        client.post.assert_called_once_with("chat/completions", {
            "model": "model", "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.2, "max_completion_tokens": 77, "stream": False,
        })

    def test_chat_invalid_response(self):
        client = Mock()
        for result in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
                       {"choices": [{"message": {"content": " "}}]}):
            client.post.return_value = result
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, "no text"):
                AlbertChat(client, "model").invoke("question")

    def test_rerank_restores_document_indices(self):
        client = Mock()
        client.post.return_value = {"results": [
            {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}
        reranker = AlbertReranker(client, "ranker")
        self.assertEqual(reranker.rerank("q", self.docs, top_n=1, return_scores=True),
                         [(self.docs[1], 0.9, 0.9, 0.9)])
        self.assertEqual(reranker.rerank("q", self.docs), [self.docs[1], self.docs[0]])
        self.assertEqual(client.post.call_args.args[1]["documents"], ["first", "second"])
        self.assertEqual(client.post.call_args.args[1]["top_n"], 2)

    def test_multi_variant_max_pooling_and_deduplication(self):
        client = Mock()
        client.post.side_effect = [
            {"results": [{"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.1}]},
            {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.3}]},
        ]
        result = AlbertReranker(client, "ranker", multi_variant_enabled=True).rerank(
            "q", self.docs, query_variants=["q", " Q ", "other", " "], return_scores=True)
        self.assertEqual(result, [(self.docs[0], 0.9, 0.1, 0.9), (self.docs[1], 0.5, 0.5, 0.5)])
        self.assertEqual(client.post.call_count, 2)

    def test_invalid_rerank_results_fail(self):
        client = Mock()
        invalid = [[], [{"index": 0, "relevance_score": 0.1}],
                   [{"index": -1, "relevance_score": 0.1}],
                   [{"index": 2, "relevance_score": 0.1}],
                   [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}],
                   [{"index": 0, "relevance_score": float("nan")}]]
        for results in invalid:
            client.post.return_value = {"results": results}
            with self.subTest(results=results), self.assertRaisesRegex(RuntimeError, "scores"):
                AlbertReranker(client, "ranker").rerank("q", self.docs)

    def test_empty_documents_skip_api(self):
        client = Mock()
        self.assertEqual(AlbertReranker(client, "ranker").rerank("q", []), [])
        client.post.assert_not_called()

    def test_ollama_startup_selection(self):
        self.assertTrue(needs_ollama({}))
        config = {"generation": {"provider": "albert"}, "query_expansion": {"rewriter": "albert"}}
        self.assertFalse(needs_ollama(config))
        config["query_expansion"] = {"rewriter": "ollama", "enabled": False}
        self.assertFalse(needs_ollama(config))
        config["query_expansion"]["enabled"] = True
        self.assertTrue(needs_ollama(config))

    def test_generator_preserves_core_prompt_without_ollama(self):
        from core.generator import AnswerGenerator
        with patch("core.generator.ChatOllama", side_effect=AssertionError("Ollama must not start")):
            generator = build_generator({"generation": {"provider": "albert", "albert_model_name": "custom"}})
        self.assertIsInstance(generator, AnswerGenerator)
        generator.llm.client = Mock()
        generator.llm.client.post.return_value = {"choices": [{"message": {"content": "answer"}}]}
        self.assertEqual(generator.answer("q", self.docs, "history"), "answer")
        payload = generator.llm.client.post.call_args.args[1]
        self.assertEqual(payload["model"], "custom")
        for text in ("first", "second", "history", "Question: q"):
            self.assertIn(text, payload["messages"][0]["content"])

    def test_query_rewriting_uses_albert_and_disabled_skips_ollama(self):
        with patch("core.query_expander.ChatOllama", side_effect=AssertionError("Ollama must not start")):
            expander = build_query_expander({"query_expansion": {"rewriter": "albert"}})
            expander.rewriter.llm.client = Mock()
            expander.rewriter.llm.client.post.return_value = {"choices": [{"message": {"content": "translated"}}]}
            self.assertEqual(expander.rewriter.translate("question", "en", "fr"), "translated")
            build_query_expander({"query_expansion": {"rewriter": "ollama", "enabled": False}})

    def test_local_generation_remains_default(self):
        with patch("core.generator.ChatOllama") as ollama:
            build_generator({"generation": {"model_name": "existing"}})
            self.assertEqual(ollama.call_args.kwargs["model"], "existing")

    def test_albert_reranker_factory_and_unknown_provider(self):
        reranker = build_reranker({"retrieval": {"rerank_provider": "albert", "albert_rerank_model_name": "custom"}})
        self.assertIsInstance(reranker, AlbertReranker)
        self.assertEqual(reranker.model_name, "custom")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_reranker({"retrieval": {"rerank_provider": "invalid"}})


if __name__ == "__main__":
    unittest.main()
