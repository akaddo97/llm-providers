# llm-providers

Lightweight Protocol-based abstraction for LLM providers — Claude (Anthropic), Gemini (Google), OpenAI. One canonical interface, three reference implementations, ~250 lines, MIT-licensed.

## What it is

Most call sites that talk to an LLM care about two things: streaming a chat (multi-turn, tool-using) and getting a sync single-shot completion. This package gives you both, with a single shape across providers, so you don't branch by name.

```python
from llm_providers import get_provider

prov = get_provider()  # reads LLM_PROVIDER env var (default: "claude")
print(prov.complete([{"role": "user", "content": "Reply with one word: ok"}]))
```

Switch provider at runtime:

```bash
LLM_PROVIDER=gemini python myscript.py
LLM_PROVIDER=openai python myscript.py
```

Or pin one explicitly:

```python
prov = get_provider("openai", model="gpt-4o-mini")
```

## Why it exists

The popular alternatives are heavy (LangChain) or commercial-tier (LiteLLM enterprise). This is a Protocol class, three implementations, and a tiny registry — readable in one sitting, copyable into a project, no plugin system.

## Install

Requires Python 3.11+. Not on PyPI yet — install from GitHub:

```bash
uv pip install git+https://github.com/akaddo97/llm-providers
# or
pip install git+https://github.com/akaddo97/llm-providers
```

**macOS users** — if your `python3` on PATH is Homebrew Python 3.13 or 3.14, `uv` may refuse with `platform.mac_ver()` returned an empty value. Use Python 3.12 explicitly via a venv:

```bash
uv venv --python /opt/homebrew/opt/python@3.12/bin/python3.12 .venv
source .venv/bin/activate
uv pip install git+https://github.com/akaddo97/llm-providers
```

Set the keys you'll use:

```bash
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```

You only need keys for the providers you actually instantiate — clients are lazy.

## Streaming chat

`Provider.chat()` yields canonical chunk dicts regardless of which provider is producing them:

```python
{"type": "text",           "text": str}
{"type": "tool_use_start", "tool": {"id", "name", "input_json": ""}}
{"type": "tool_use_input", "partial_json": str}
{"type": "tool_use_end",   "tool": {"id", "name", "input": {...parsed...}}}
{"type": "stop",           "stop_reason": str, "usage": {input_tokens, output_tokens, ...}}
```

This means a route can forward the stream to a browser as Server-Sent Events without knowing which provider is upstream. `tool_use_end` carries the fully-parsed input so the route can execute the tool without rebuilding JSON.

```python
prov = get_provider("claude")
for chunk in prov.chat(
    messages=[{"role": "user", "content": "what's the weather in Paris?"}],
    system="you are concise",
    tools=[{
        "name": "get_weather",
        "description": "Look up current weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }],
):
    if chunk["type"] == "text":
        print(chunk["text"], end="", flush=True)
    elif chunk["type"] == "tool_use_end":
        # Execute the tool with chunk["tool"]["input"], feed result back.
        ...
```

Tool definitions use the Anthropic shape `{name, description, input_schema}`. Each provider translates internally.

## Single-shot completion

`Provider.complete()` is the sync, no-streaming, no-tools shape — for cleanup, classification, summarisation, anywhere you want a string back:

```python
prov = get_provider()
text = prov.complete(
    messages=[{"role": "user", "content": "Summarise: ..."}],
    system="reply in one sentence",
    max_tokens=200,
    temperature=0.0,
)
```

## Architecture

- `Provider` — `Protocol` class declaring `chat()` + `complete()` + `name` + `model`.
- `ClaudeProvider`, `GeminiProvider`, `OpenAIProvider` — concrete implementations. Each translates the canonical surface to its native SDK shape.
- `get_provider(name=None, **kwargs)` — registry-backed factory. `name=None` reads `LLM_PROVIDER` env var with a `claude` fallback. Extra kwargs (`model`, `api_key`) flow through to the provider constructor.
- `default_provider_name()` — the helper. Read it instead of hard-coding a provider string anywhere.
- All clients are **lazy**: missing API keys are fine at construction; the error surfaces only when you call `chat()` / `complete()`.

System prompt translation, message-role translation (`assistant` → `model` for Gemini, system message prepending for OpenAI), and tool-schema translation all happen inside the provider. Callers see one shape.

## Limits

- **Gemini tool-use:** v0.1 supports text-only streaming on Gemini. `chat()` with `tools=[...]` raises `NotImplementedError`. Route tool-using sites to Claude or OpenAI for now.
- **Tool-result feedback:** the chunk contract describes what the model emits; threading tool *results* back into a follow-up turn is the caller's responsibility (and currently provider-shaped — Anthropic-style content blocks).
- **Cost / retries / streaming-resumption:** none of these are wrapped. The package stays out of the way.

## Versioning

`0.1.x` — API may shift between minor versions until `1.0`. Pin if you depend on it.

## License

MIT. See [LICENSE](LICENSE).
