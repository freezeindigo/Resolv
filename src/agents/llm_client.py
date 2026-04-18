"""
Unified async LLM calls — Groq (default) or Anthropic (arbiter only).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import anthropic
from groq import AsyncGroq

from src.config.model_config import ANTHROPIC_API_KEY, GROQ_API_KEY, MODEL_CONFIG

_groq_client: Optional[AsyncGroq] = None
_anthropic_client: Optional[anthropic.AsyncAnthropic] = None


def _groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (required for Groq-backed roles)")
        _groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _groq_client


def _anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


async def llm_call(
    role: str,
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    role: tier2_reasoning | hypothesis_agents | pattern_interpreter | arbiter | judge

    Returns:
        text, tokens, latency_ms, model, provider
    """
    if role not in MODEL_CONFIG:
        raise ValueError(f"Unknown LLM role: {role}")

    cfg = MODEL_CONFIG[role]
    provider = cfg["provider"]
    model = cfg["model"]
    mt = max_tokens if max_tokens is not None else int(cfg.get("max_tokens", 800))

    t0 = time.monotonic()

    if provider == "groq":
        client = _groq()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=mt,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = (response.choices[0].message.content or "").strip()
        us = response.usage
        tokens = 0
        if us is not None:
            tokens = getattr(us, "total_tokens", None) or (
                (getattr(us, "prompt_tokens", None) or 0)
                + (getattr(us, "completion_tokens", None) or 0)
            )
        return {
            "text": text,
            "tokens": int(tokens or 0),
            "latency_ms": latency_ms,
            "model": model,
            "provider": "groq",
        }

    # Anthropic
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (required for arbiter)")
    client = _anthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=mt,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        "text": text,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "model": model,
        "provider": "anthropic",
    }
