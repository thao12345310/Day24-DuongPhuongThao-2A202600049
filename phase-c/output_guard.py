"""Output guardrails for Lab 24.

Groq Llama Guard 3/4 has been decommissioned. Uses llama-3.1-8b-instant on
Groq as an LLM safety judge with the Llama Guard 3 category taxonomy.
Requires GROQ_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import requests


_SAFETY_SYSTEM = (
    "You are a content safety classifier. Evaluate the ASSISTANT RESPONSE for unsafe content. "
    "Unsafe categories: S1=Violent Crimes, S2=Non-Violent Crimes, S3=Sex-Related Crimes, "
    "S4=Child Safety, S5=Indiscriminate Weapons, S6=Hate Speech, "
    "S7=Suicide/Self-Harm, S8=Malware/Hacking/Fraud. "
    "Reply with ONLY 'safe' OR 'unsafe S<N>' on a single line. No other text."
)


@dataclass(frozen=True)
class OutputGuardResult:
    safe: bool
    label: str
    latency_ms: float


class LlamaGuard3OutputGuard:
    """LLM safety judge via Groq API (requires GROQ_API_KEY).

    Uses llama-3.1-8b-instant with the Llama Guard 3 category taxonomy as a
    drop-in replacement after Groq decommissioned llama-guard-3-8b (2025-06).
    """

    def __init__(self, api_key: str | None = None, timeout_s: float = 15.0) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required for LlamaGuard3OutputGuard. "
                "Set the environment variable before instantiating this class."
            )
        self.timeout_s = timeout_s
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        start = time.perf_counter()
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": _SAFETY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"User query: {user_input or 'N/A'}\n\n"
                        f"Assistant response to evaluate: {agent_response}"
                    ),
                },
            ],
            "max_tokens": 30,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
        if not resp.ok:
            raise RuntimeError(f"Groq API {resp.status_code}: {resp.text}")
        label = resp.json()["choices"][0]["message"]["content"].strip()
        safe = label.lower().startswith("safe")
        latency_ms = (time.perf_counter() - start) * 1000
        return safe, label, latency_ms

    async def check_async(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        return await asyncio.to_thread(self.check, user_input, agent_response)
