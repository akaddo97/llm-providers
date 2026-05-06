"""Tests for get_provider() registry kwargs pass-through.

Complements test_provider_abstraction.py, which covers the registry's
default-name + unknown-name behaviour. This file pins the kwargs path:
get_provider("claude", model=...) must construct ClaudeProvider with that
model, and similarly for the other two providers.
"""
from __future__ import annotations

import pytest

import llm_providers as p


def test_get_provider_claude_passes_model_kwarg(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    prov = p.get_provider("claude", model="claude-sonnet-4-6")
    assert prov.name == "claude"
    assert prov.model == "claude-sonnet-4-6"


def test_get_provider_gemini_passes_model_kwarg():
    prov = p.get_provider("gemini", model="gemini-2.5-flash")
    assert prov.name == "gemini"
    assert prov.model == "gemini-2.5-flash"


def test_get_provider_openai_passes_model_kwarg():
    prov = p.get_provider("openai", model="gpt-4o-mini")
    assert prov.name == "openai"
    assert prov.model == "gpt-4o-mini"


def test_get_provider_passes_explicit_api_key():
    prov = p.get_provider("claude", api_key="sk-explicit")
    # Internal field — but we check it's stored, not validated yet (lazy client).
    assert prov._api_key == "sk-explicit"


def test_get_provider_unknown_lists_known_names():
    with pytest.raises(ValueError, match="claude"):
        p.get_provider("nonexistent_xyz")
