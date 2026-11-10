"""Provider-agnostic LLM client for free / local models.

Every supported provider speaks the OpenAI "chat completions" protocol, so a
single HTTP client covers them all. No paid SDK is required.

Keys are read (in order) from: explicit argument -> Streamlit secrets ->
environment variable. When no key is configured the app falls back to the
rule-based engine, so it always works offline.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

# Model IDs on free tiers change often. They are defaults only and can be
# overridden in the app's Settings tab or via <PROVIDER>_MODEL env vars.
PROVIDERS: dict[str, dict[str, str]] = {
    "Groq (free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "signup": "https://console.groq.com/keys",
    },
    "Google Gemini (free tier)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "signup": "https://aistudio.google.com/apikey",
    },
    "OpenRouter (free models)": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "openrouter/free",
        "signup": "https://openrouter.ai/keys",
    },
    "Ollama (local, no key)": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "llama3.2",
        "signup": "https://ollama.com/download",
    },
}


class LLMError(RuntimeError):
    pass


def _secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # imported lazily so the module works outside Streamlit

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


@dataclass
class LLMClient:
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 45

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {self.provider}")
        cfg = PROVIDERS[self.provider]
        self.base_url = (_secret(cfg["env_key"].replace("_API_KEY", "_BASE_URL")) or cfg["base_url"]).rstrip("/")
        self.api_key = self.api_key or _secret(cfg["env_key"])
        self.model = self.model or _secret(cfg["env_key"].replace("_API_KEY", "_MODEL")) or cfg["default_model"]

    @property
    def ready(self) -> bool:
        return bool(self.api_key) or self.provider.startswith("Ollama")

    def chat(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 1200) -> str:
        if not self.ready:
            raise LLMError("No API key configured")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f"Network error: {e}") from e
        if r.status_code == 429:
            raise LLMError("Rate limit reached on the free tier; wait a minute or switch provider")
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected response: {r.text[:300]}") from e

    def chat_json(self, system: str, user: str, **kw: Any) -> dict:
        """Ask for JSON and parse it robustly (models sometimes wrap it in prose or fences)."""
        raw = self.chat(system + "\nRespond with ONE valid JSON object and nothing else.", user, **kw)
        return parse_json_loose(raw)


def parse_json_loose(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("Model did not return JSON")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e}") from e


def get_client(provider: Optional[str], api_key: Optional[str] = None, model: Optional[str] = None) -> Optional[LLMClient]:
    """Return a ready client, or None when the app should use the rule-based engine."""
    if not provider or provider == "Off (rule-based only)":
        return None
    client = LLMClient(provider, api_key or None, model or None)
    return client if client.ready else None
