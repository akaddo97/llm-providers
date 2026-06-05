"""Tests for OpenAICompatibleProvider + the get_provider() presets.

The provider subclasses OpenAIProvider and reuses its wire translation, so
these tests pin only the parts that differ: client construction (base_url +
key source), the preset registry (deepseek / ollama / ...), and the generic
"openai-compatible" path. The shared chat()/complete() translation is already
covered by the OpenAIProvider tests.
"""
from __future__ import annotations

import pytest

import llm_providers as p


def _fake_openai(captured: dict):
    """An openai.OpenAI stand-in that records constructor kwargs and exposes a
    chat.completions.create returning a fixed assistant message."""

    class _Msg:
        content = "hi from the endpoint"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kw):
            captured.setdefault("calls", []).append(kw)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kw):
            captured.update(kw)
            self.chat = _Chat()

    return _Client


# --- preset registry ---


def test_deepseek_preset_sets_base_url_model_and_name():
    prov = p.get_provider("deepseek")
    assert isinstance(prov, p.OpenAICompatibleProvider)
    assert prov.name == "deepseek"
    assert prov.model == "deepseek-chat"
    assert prov.base_url == "https://api.deepseek.com"


def test_deepseek_preset_model_override_keeps_name():
    prov = p.get_provider("deepseek", model="deepseek-reasoner")
    assert prov.model == "deepseek-reasoner"
    assert prov.name == "deepseek"


def test_ollama_preset_requires_model_then_works():
    with pytest.raises(ValueError, match="model"):
        p.get_provider("ollama")  # local preset has no default model
    prov = p.get_provider("ollama", model="llama3.1")
    assert prov.base_url == "http://localhost:11434/v1"
    assert prov.model == "llama3.1"
    assert prov.name == "ollama"


def test_generic_openai_compatible_requires_base_url_and_model():
    with pytest.raises(ValueError, match="base_url"):
        p.get_provider("openai-compatible", model="x")
    with pytest.raises(ValueError, match="model"):
        p.get_provider("openai-compatible", base_url="http://localhost:8000/v1")
    prov = p.get_provider(
        "openai-compatible", base_url="http://localhost:8000/v1/", model="my-model"
    )
    assert prov.base_url == "http://localhost:8000/v1"  # trailing slash stripped
    assert prov.name == "openai-compatible"


def test_unknown_provider_lists_presets():
    with pytest.raises(ValueError, match="deepseek"):
        p.get_provider("nonexistent_zzz")


# --- client construction ---


def test_client_built_with_base_url_and_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", _fake_openai(captured))
    p.get_provider("deepseek")._ensure_client()
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["api_key"] == "sk-deepseek-test"


def test_local_backend_uses_placeholder_key_when_env_blank(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", _fake_openai(captured))
    p.get_provider("ollama", model="llama3.1")._ensure_client()
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"]  # non-empty placeholder, never None


def test_explicit_api_key_beats_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", _fake_openai(captured))
    p.get_provider("deepseek", api_key="sk-explicit")._ensure_client()
    assert captured["api_key"] == "sk-explicit"


# --- inherited wire translation still works through the subclass ---


def test_complete_routes_through_openai_wire(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured: dict = {}
    monkeypatch.setattr("openai.OpenAI", _fake_openai(captured))
    prov = p.get_provider("deepseek")
    out = prov.complete(
        [{"role": "user", "content": "hello"}], system="be brief"
    )
    assert out == "hi from the endpoint"
    call = captured["calls"][0]
    assert call["model"] == "deepseek-chat"
    # system folded into a leading system-role message (OpenAI shape)
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["messages"][1] == {"role": "user", "content": "hello"}
