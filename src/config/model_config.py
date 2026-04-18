"""LLM provider + model id per pipeline role (see src/agents/llm_client.py)."""

import os

MODEL_CONFIG = {
    "tier2_reasoning": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "max_tokens": 800,
    },
    "hypothesis_agents": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "max_tokens": 1024,
    },
    "pattern_interpreter": {
        "provider": "groq",
        "model": "llama-3.1-70b-versatile",
        "max_tokens": 600,
    },
    "arbiter": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
    },
    "judge": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "max_tokens": 600,
    },
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
