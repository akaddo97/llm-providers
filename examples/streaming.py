"""Stream a response token-by-token using the canonical chunk contract.

Demonstrates the chat() loop the README describes in prose: dispatch on
chunk["type"] and ignore the rest. Run with one key set:

    export ANTHROPIC_API_KEY=...
    python examples/streaming.py

Or pin a specific provider:

    LLM_PROVIDER=openai python examples/streaming.py
"""
from __future__ import annotations

from llm_providers import get_provider


def main() -> None:
    prov = get_provider()
    print(f"[provider: {prov.name} | model: {prov.model}]")

    usage = None
    for chunk in prov.chat(
        messages=[{"role": "user", "content": "Hello, tell me one sentence about Python."}],
        system="be concise",
    ):
        if chunk["type"] == "text":
            print(chunk["text"], end="", flush=True)
        elif chunk["type"] == "stop":
            usage = chunk["usage"]

    print()
    if usage:
        print(f"[usage: in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}]")


if __name__ == "__main__":
    main()
