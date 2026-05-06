"""Drop-in switching across Claude / Gemini / OpenAI via env var.

Run with one or more keys set:

    export ANTHROPIC_API_KEY=...
    python examples/quickstart.py

Or pin a specific provider via env var:

    LLM_PROVIDER=gemini python examples/quickstart.py
"""
from __future__ import annotations

from llm_providers import get_provider


def main() -> None:
    # Default — reads LLM_PROVIDER env var (falls back to "claude").
    prov = get_provider()
    print(f"[default provider: {prov.name} | model: {prov.model}]")
    print(prov.complete([{"role": "user", "content": "Reply with one word: ok"}]))

    # Explicit override at construction.
    prov = get_provider("openai", model="gpt-4o-mini")
    print(f"[explicit provider: {prov.name} | model: {prov.model}]")
    print(prov.complete([{"role": "user", "content": "Reply with one word: ok"}]))


if __name__ == "__main__":
    main()
