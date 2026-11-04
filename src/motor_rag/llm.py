"""Groq client (OpenAI-compatible API). Key and model come from .env."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@lru_cache
def client() -> OpenAI:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    # The SDK retries 429 rate limits and 5xx errors with exponential backoff.
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL, max_retries=5, timeout=60)


def chat(messages: list[dict], **kwargs):
    return client().chat.completions.create(model=model_name(), temperature=0, messages=messages, **kwargs)
