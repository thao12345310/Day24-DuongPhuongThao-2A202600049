"""Shared OpenAI helpers for Lab 24 scripts.

The lab runners intentionally use the official OpenAI SDK directly for the
RAG pipeline and LLM judge paths.  LangChain/RAGAS integrations are still used
where the lab asks for them, but this module keeps API-key loading, JSON
parsing, and token-cost accounting in one place.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHAT_MODEL = os.getenv("LAB24_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("LAB24_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


MODEL_PRICES_USD_PER_1M_TOKENS = {
    # Public GPT-4o mini prices used by the lab era examples. Override with
    # LAB24_INPUT_PRICE_PER_1M and LAB24_OUTPUT_PRICE_PER_1M if your account
    # pricing differs.
    "gpt-4o-mini": (0.15, 0.60),
}


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""

    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_openai_client():
    """Return an OpenAI client after loading .env.

    Import is local so offline deterministic scripts can still run when the SDK
    has not been installed yet.
    """

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it before running this script.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The official OpenAI SDK is missing. Run: pip install -r requirements.txt") from exc
    return OpenAI()


def extract_json(text: str) -> Any:
    """Parse JSON from strict JSON output or a fenced response."""

    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


@dataclass
class CostTracker:
    """Approximate OpenAI cost logger for README/reporting."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add_usage(self, usage: Any) -> None:
        if not usage:
            return
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )

    def estimate_usd(self, model: str = DEFAULT_CHAT_MODEL) -> float:
        input_price = float(os.getenv("LAB24_INPUT_PRICE_PER_1M", "nan"))
        output_price = float(os.getenv("LAB24_OUTPUT_PRICE_PER_1M", "nan"))
        if input_price != input_price or output_price != output_price:
            input_price, output_price = MODEL_PRICES_USD_PER_1M_TOKENS.get(model, (0.0, 0.0))
        return (self.input_tokens / 1_000_000 * input_price) + (self.output_tokens / 1_000_000 * output_price)

    def as_dict(self, model: str = DEFAULT_CHAT_MODEL) -> dict[str, float | int | str]:
        return {
            "model": model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimate_usd(model), 6),
        }
