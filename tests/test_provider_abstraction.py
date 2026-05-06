"""Tests for the provider abstraction.

Covers:
- Protocol conformance for all three providers (chat + complete signatures).
- Each provider's complete() with mocked SDK clients.
- Each provider's chat() text streaming with mocked SDK streams.
- OpenAI tool-use streaming → canonical chunk shapes (translation).
- get_provider() registry, env-var default, unknown name.
- Lazy client init: missing key OK at construction, RuntimeError at call.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import llm_providers as p


# --- Protocol conformance ---


def test_all_providers_expose_protocol_surface():
    for name in ("claude", "gemini", "openai"):
        prov = p.get_provider(name)
        assert prov.name == name
        assert isinstance(prov.model, str) and prov.model
        assert callable(prov.chat)
        assert callable(prov.complete)


def test_default_provider_name_uses_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert p.default_provider_name() == "openai"


def test_default_provider_name_falls_back_to_claude(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert p.default_provider_name() == "claude"


def test_get_provider_no_name_uses_env_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    prov = p.get_provider()
    assert prov.name == "gemini"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        p.get_provider("nonexistent_xyz")


# --- ClaudeProvider ---


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(type="text", text=text)]


class _FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.last_kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeAnthropicResponse("polished output")


def test_claude_complete_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = _FakeAnthropicClient()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)
    prov = p.ClaudeProvider(model="claude-haiku-4-5-20251001")
    result = prov.complete(
        messages=[{"role": "user", "content": "hello"}],
        system="be concise",
        max_tokens=128,
        temperature=0.0,
    )
    assert result == "polished output"
    assert fake.last_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert fake.last_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert fake.last_kwargs["system"] == "be concise"
    assert fake.last_kwargs["max_tokens"] == 128
    assert fake.last_kwargs["temperature"] == 0.0


def test_claude_complete_omits_temperature_when_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = _FakeAnthropicClient()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)
    prov = p.ClaudeProvider()
    prov.complete(messages=[{"role": "user", "content": "hi"}])
    assert "temperature" not in fake.last_kwargs


def test_claude_complete_multi_turn_messages_pass_through(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = _FakeAnthropicClient()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)
    prov = p.ClaudeProvider()
    msgs = [
        {"role": "user", "content": "draft 1"},
        {"role": "assistant", "content": "polished 1"},
        {"role": "user", "content": "draft 2"},
    ]
    prov.complete(messages=msgs)
    assert fake.last_kwargs["messages"] == msgs


def test_claude_complete_no_system_omits_field(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = _FakeAnthropicClient()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)
    prov = p.ClaudeProvider()
    prov.complete(messages=[{"role": "user", "content": "hi"}])
    assert "system" not in fake.last_kwargs


# --- GeminiProvider ---


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeGeminiModels:
    def __init__(self):
        self.last_kwargs = None
        self.last_stream_kwargs = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeGeminiResponse("gemini said this")

    def generate_content_stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        yield SimpleNamespace(text="hello ", usage_metadata=None, candidates=[])
        yield SimpleNamespace(text="world", usage_metadata=None, candidates=[])
        yield SimpleNamespace(
            text=None,
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )


class _FakeGeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = _FakeGeminiModels()


def test_gemini_complete_returns_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    fake = _FakeGeminiClient()
    monkeypatch.setattr("google.genai.Client", lambda **kw: fake)
    prov = p.GeminiProvider(model="gemini-2.5-flash")
    result = prov.complete(
        messages=[{"role": "user", "content": "hello"}],
        system="be concise",
        max_tokens=256,
        temperature=0.2,
    )
    assert result == "gemini said this"
    assert fake.models.last_kwargs["model"] == "gemini-2.5-flash"
    config = fake.models.last_kwargs["config"]
    # GenerateContentConfig is a Pydantic-style model; access fields by attr.
    assert config.system_instruction == "be concise"
    assert config.max_output_tokens == 256
    assert config.temperature == 0.2


def test_gemini_complete_translates_assistant_to_model_role(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    fake = _FakeGeminiClient()
    monkeypatch.setattr("google.genai.Client", lambda **kw: fake)
    prov = p.GeminiProvider()
    prov.complete(messages=[
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    contents = fake.models.last_kwargs["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "a1"}]


def test_gemini_chat_text_streaming_yields_canonical_chunks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    fake = _FakeGeminiClient()
    monkeypatch.setattr("google.genai.Client", lambda **kw: fake)
    prov = p.GeminiProvider()
    chunks = list(prov.chat(
        messages=[{"role": "user", "content": "hi"}],
        system="be brief",
    ))
    text_chunks = [c for c in chunks if c["type"] == "text"]
    stop_chunks = [c for c in chunks if c["type"] == "stop"]
    assert "".join(c["text"] for c in text_chunks) == "hello world"
    assert len(stop_chunks) == 1
    assert stop_chunks[0]["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_gemini_chat_with_tools_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    prov = p.GeminiProvider()
    with pytest.raises(NotImplementedError, match="does not yet support tools"):
        list(prov.chat(
            messages=[],
            system="",
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
        ))


# --- OpenAIProvider ---


class _FakeOpenAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content):
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, content):
        self.choices = [_FakeOpenAIChoice(content)]


class _FakeOpenAICompletions:
    def __init__(self, stream_chunks=None):
        self.last_kwargs = None
        self._stream_chunks = stream_chunks or []

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self._stream_chunks)
        return _FakeOpenAIResponse("openai said this")


class _FakeOpenAIClient:
    def __init__(self, api_key=None, stream_chunks=None):
        self.api_key = api_key
        completions = _FakeOpenAICompletions(stream_chunks=stream_chunks)
        self.chat = SimpleNamespace(completions=completions)


def test_openai_complete_returns_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = _FakeOpenAIClient()
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    prov = p.OpenAIProvider(model="gpt-4o-mini")
    result = prov.complete(
        messages=[{"role": "user", "content": "hello"}],
        system="be concise",
        max_tokens=128,
        temperature=0.0,
    )
    assert result == "openai said this"
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"][0] == {"role": "system", "content": "be concise"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hello"}


def test_openai_complete_omits_system_message_when_blank(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = _FakeOpenAIClient()
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    prov = p.OpenAIProvider()
    prov.complete(messages=[{"role": "user", "content": "hi"}])
    msgs = fake.chat.completions.last_kwargs["messages"]
    assert all(m["role"] != "system" for m in msgs)


def _openai_text_chunk(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, tool_calls=None),
            finish_reason=None,
        )],
        usage=None,
    )


def _openai_finish_chunk(reason="stop", input_tokens=10, output_tokens=4):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None),
            finish_reason=reason,
        )],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def test_openai_chat_text_streaming_yields_canonical_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chunks_in = [
        _openai_text_chunk("hello "),
        _openai_text_chunk("world"),
        _openai_finish_chunk(),
    ]
    fake = _FakeOpenAIClient(stream_chunks=chunks_in)
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    prov = p.OpenAIProvider()
    out = list(prov.chat(messages=[{"role": "user", "content": "hi"}], system="be brief"))
    text = "".join(c["text"] for c in out if c["type"] == "text")
    assert text == "hello world"
    stop = [c for c in out if c["type"] == "stop"]
    assert len(stop) == 1
    assert stop[0]["stop_reason"] == "end_turn"
    assert stop[0]["usage"] == {"input_tokens": 10, "output_tokens": 4}


def test_openai_chat_with_tools_translates_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = _FakeOpenAIClient(stream_chunks=[_openai_finish_chunk()])
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    prov = p.OpenAIProvider()
    list(prov.chat(
        messages=[{"role": "user", "content": "x"}],
        system="",
        tools=[{
            "name": "search",
            "description": "search the graph",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }],
    ))
    sent_tools = fake.chat.completions.last_kwargs["tools"]
    assert sent_tools == [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "search the graph",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }]


def test_openai_chat_tool_use_streaming_yields_canonical_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Simulate OpenAI's streaming tool-call deltas:
    # first delta gives id+name+empty args; subsequent deltas stream args JSON.
    tc_first = SimpleNamespace(
        index=0,
        id="call_abc",
        function=SimpleNamespace(name="search", arguments=""),
    )
    tc_args1 = SimpleNamespace(
        index=0, id=None,
        function=SimpleNamespace(name=None, arguments='{"q":'),
    )
    tc_args2 = SimpleNamespace(
        index=0, id=None,
        function=SimpleNamespace(name=None, arguments='"hello"}'),
    )
    chunks_in = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc_first]),
                finish_reason=None,
            )],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc_args1]),
                finish_reason=None,
            )],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc_args2]),
                finish_reason=None,
            )],
            usage=None,
        ),
        _openai_finish_chunk(reason="tool_calls"),
    ]
    fake = _FakeOpenAIClient(stream_chunks=chunks_in)
    monkeypatch.setattr("openai.OpenAI", lambda **kw: fake)
    prov = p.OpenAIProvider()
    out = list(prov.chat(
        messages=[{"role": "user", "content": "x"}],
        system="",
        tools=[{"name": "search", "description": "d", "input_schema": {}}],
    ))
    starts = [c for c in out if c["type"] == "tool_use_start"]
    inputs = [c for c in out if c["type"] == "tool_use_input"]
    ends = [c for c in out if c["type"] == "tool_use_end"]
    stops = [c for c in out if c["type"] == "stop"]
    assert len(starts) == 1
    assert starts[0]["tool"]["name"] == "search"
    assert starts[0]["tool"]["id"] == "call_abc"
    # Concatenated input deltas reconstruct the JSON.
    assert "".join(c["partial_json"] for c in inputs) == '{"q":"hello"}'
    assert len(ends) == 1
    assert ends[0]["tool"]["input"] == {"q": "hello"}
    assert stops[0]["stop_reason"] == "tool_use"


# --- lazy client + missing key behaviour ---


def test_claude_missing_key_at_construction_does_not_raise(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Construction is safe — only the call should raise.
    prov = p.ClaudeProvider()
    assert prov.name == "claude"


def test_claude_missing_key_at_call_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prov = p.ClaudeProvider()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        prov.complete(messages=[{"role": "user", "content": "hi"}])


def test_gemini_missing_key_at_call_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    prov = p.GeminiProvider()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not set"):
        prov.complete(messages=[{"role": "user", "content": "hi"}])


def test_openai_missing_key_at_call_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prov = p.OpenAIProvider()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        prov.complete(messages=[{"role": "user", "content": "hi"}])


def test_explicit_api_key_arg_overrides_missing_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = _FakeAnthropicClient()
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: fake)
    prov = p.ClaudeProvider(api_key="sk-explicit")
    result = prov.complete(messages=[{"role": "user", "content": "hi"}])
    assert result == "polished output"
