"""Groq client (OpenAI-compatible API). Key and models come from .env."""
import os
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Tokens used per model in this process. The free tier allows a fixed number of
# tokens per model per day, so evals report cost alongside quality.
usage: Counter = Counter()


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def judge_model_name() -> str:
    # A different model from the one being graded: avoids a model grading its own
    # answers, and on Groq each model has its own daily token quota.
    return os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")


@lru_cache
def client() -> OpenAI:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    # The SDK retries 429 rate limits and 5xx errors with exponential backoff.
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL, max_retries=5, timeout=60)


def chat(messages: list[dict], model: str | None = None, **kwargs):
    model = model or model_name()
    resp = client().chat.completions.create(model=model, temperature=0, messages=messages, **kwargs)
    if resp.usage:
        usage[model] += resp.usage.total_tokens
    return resp
