# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `OpenAICompatibleProvider` — one adapter for any OpenAI-compatible `/chat/completions` endpoint (DeepSeek, Ollama, vLLM, llama.cpp, Groq, Together, OpenRouter, Fireworks, Mistral, or a custom gateway). Subclasses `OpenAIProvider` and swaps only the client `base_url` + key source, inheriting the request / streaming / tool-use translation unchanged.
- `get_provider()` presets: `deepseek`, `ollama`, `groq`, `together`, `openrouter`, `fireworks`, `mistral` (each fills in base_url + key env), plus `openai-compatible` / `custom` for a hand-supplied `base_url`. `LLM_PROVIDER=deepseek` makes DeepSeek the process default.
- `DEEPSEEK_DEFAULT_MODEL` constant (`deepseek-chat`), exported alongside the other per-provider defaults.
- `temperature` keyword threaded through `chat()` on all three providers (previously `complete()`-only).
- `__version__` exposed on the package via `importlib.metadata`, with a fallback to the in-tree constant for editable / source-tree imports.
- Provider capability matrix in the README (streaming, tool-use, prompt-caching support per provider).
- `examples/streaming.py` — chunk-dispatch loop demo, companion to `examples/quickstart.py`.
- Tests / Python 3.11+ / MIT badges in the README hero.

### Changed
- README install instructions lead with `pip install` (broadest baseline) before the `uv pip install` alternate.
- SDK dependency upper bounds capped (`anthropic<1`, `google-genai<2`, `openai<2`) so 0.1.x users aren't broken by upstream major bumps.
- Author email switched to a GitHub noreply alias in `pyproject.toml`.
- `Repository` URL surfaced in `[project.urls]` alongside `Homepage` and `Issues`.

### Fixed
- OpenAI tool-use streaming: `tool_use_input` chunks are now gated on `_started`, and pre-name argument fragments flushed at start, so callers don't receive `partial_json` deltas before the tool's identity is known.

### Removed
- Dead `if/else` branch in `ClaudeProvider.complete()` that assigned the same value in both arms — the Anthropic SDK accepts both `str` and block-list system prompts natively.

## [0.1.0] — 2026-05-06

### Added
- Initial extract from internal tooling: `Provider` Protocol declaring `chat()` + `complete()` + `name` + `model`.
- `ClaudeProvider`, `GeminiProvider`, `OpenAIProvider` — concrete implementations translating the canonical surface to each native SDK shape.
- `get_provider(name=None, **kwargs)` registry-backed factory; `default_provider_name()` helper reading `LLM_PROVIDER` env var with a `claude` fallback.
- Streaming chunk contract: `text`, `tool_use_start`, `tool_use_input`, `tool_use_end`, `stop`. `tool_use_end` carries fully-parsed input so callers don't rebuild JSON.
- Lazy clients across all three providers — missing API keys are OK at construction; `RuntimeError` only at `chat()` / `complete()` call time.
- Anthropic ephemeral prompt-cache on `chat()` system block.
- pytest suite covering protocol conformance, complete() / chat() per provider, OpenAI tool-use streaming translation, lazy client behaviour, missing-key error messages.
- GitHub Actions CI matrix on Python 3.11 / 3.12 / 3.13.
- MIT license.
