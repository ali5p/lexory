"""Tests for LLM provider factory and Groq adapter."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm.factory import build_llm
from llm.groq_adapter import GroqAdapter


def test_build_llm_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with patch("llm.factory.OllamaAdapter") as ollama_cls:
        ollama_cls.return_value = MagicMock()
        llm = build_llm()
    assert llm is ollama_cls.return_value
    ollama_cls.assert_called_once()


def test_build_llm_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    with patch("llm.factory.GroqAdapter") as groq_cls:
        groq_cls.return_value = MagicMock()
        llm = build_llm()
    assert llm is groq_cls.return_value
    groq_cls.assert_called_once()


def test_build_llm_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        build_llm()


def test_groq_adapter_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqAdapter()


def test_groq_chat_parses_completion(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    adapter = GroqAdapter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"topic":"T","lesson":"L","exercise":"E"}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("llm.groq_adapter.requests.post", return_value=mock_response) as post:
        text = adapter.chat(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
        )

    assert "topic" in text
    post.assert_called_once()
    body = post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_groq_chat_retries_without_json_format_on_400(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    adapter = GroqAdapter()

    bad = MagicMock()
    bad.status_code = 400
    bad.raise_for_status = MagicMock()

    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"choices": [{"message": {"content": "plain"}}]}
    ok.raise_for_status = MagicMock()

    with patch("llm.groq_adapter.requests.post", side_effect=[bad, ok]) as post:
        text = adapter.chat(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
        )

    assert text == "plain"
    assert post.call_count == 2
    assert "response_format" not in post.call_args_list[1].kwargs["json"]
