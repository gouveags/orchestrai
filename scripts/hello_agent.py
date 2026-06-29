#!/usr/bin/env python3
"""Send a small hello prompt through the orchestrai Python API."""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1-mini",
}

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the orchestrai Python API with a real provider."
    )
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS),
        default=os.environ.get("ORCHESTRAI_PROVIDER", "anthropic"),
        help="Provider to use. Defaults to ORCHESTRAI_PROVIDER or anthropic.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ORCHESTRAI_MODEL"),
        help="Model to use. Defaults to ORCHESTRAI_MODEL or the provider smoke-test model.",
    )
    parser.add_argument(
        "--message",
        default="Hello",
        help="Message to send to the agent.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum output tokens for the smoke response.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    env_key = PROVIDER_ENV_KEYS[args.provider]

    if not os.environ.get(env_key):
        print(f"{env_key} is required to run this smoke test.", file=sys.stderr)
        return 2

    try:
        import orchestrai
    except ImportError as error:
        print(
            "Could not import orchestrai. Build/install the Python bindings first, "
            "for example with `uvx maturin develop --features python-extension`.",
            file=sys.stderr,
        )
        print(f"Import error: {error}", file=sys.stderr)
        return 3

    agent = orchestrai.create_agent(
        model=model,
        provider=args.provider,
        instructions="Reply briefly and directly.",
        max_tool_rounds=0,
        max_tokens=args.max_tokens,
    )
    response = agent.run(args.message)

    print(f"provider={args.provider}")
    print(f"model={model}")
    print(f"message={args.message}")
    print(f"response={response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
