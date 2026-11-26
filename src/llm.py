"""Minimal, provider-agnostic LLM client.

Set in .env (or Streamlit secrets):
    LLM_PROVIDER = anthropic | openai | ollama
    LLM_MODEL    = model name (optional; sensible default per provider)
    ANTHROPIC_API_KEY / OPENAI_API_KEY   (for those providers)
    OPENAI_BASE_URL  (optional) - any OpenAI-compatible API, e.g. Google
                     Gemini's free tier or Groq
    OLLAMA_HOST      (optional) - default http://localhost:11434

Ollama runs models locally for free: install from https://ollama.com and run
`ollama pull qwen2.5-coder:7b`.
"""
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5-coder:7b",
}


def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").strip().lower()
    if p not in DEFAULT_MODELS:
        raise RuntimeError(
            "Set LLM_PROVIDER in .env to one of: anthropic, openai, ollama "
            "(see .env.example).")
    return p


def model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider()]


def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """Send one system + user message and return the reply text."""
    p, model = provider(), model_name()

    if p == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(block.text for block in resp.content if block.type == "text")

    if p == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""

    import requests  # ollama
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    resp = requests.post(f"{host}/api/chat", timeout=180, json={
        "model": model, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    resp.raise_for_status()
    return resp.json()["message"]["content"]
