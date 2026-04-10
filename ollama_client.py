import json
import httpx
from typing import Iterator

import config
from config import OLLAMA_URL

# Persistent connection pool — eliminates TCP handshake overhead on every call.
# Small per-call gain (~20-50 ms) that compounds across many tool iterations.
_client = httpx.Client(timeout=300.0)


def chat(
    messages: list,
    tools: list,
    stream: bool = False,
    fmt: dict | str | None = None,
    model: str | None = None,
) -> dict | Iterator[dict]:
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model or config.MODEL,
        "messages": messages,
        "tools": tools,
        "stream": stream,
        "options": {"num_ctx": config.NUM_CTX},
    }
    if fmt is not None:
        payload["format"] = fmt

    # Keep the model loaded in RAM indefinitely between requests.
    # Without this Ollama unloads the model after 5 min idle, causing
    # a cold-load penalty on the next call.
    payload["keep_alive"] = -1

    if stream:
        return _stream_chat(url, payload)
    else:
        resp = _client.post(url, json=payload, timeout=300)
        _raise_friendly(resp, payload["model"])
        return resp.json()


def _raise_friendly(resp: httpx.Response, model: str) -> None:
    """Raise a readable error for common Ollama failures instead of a raw httpx traceback."""
    if resp.is_success:
        return
    if resp.status_code == 500:
        # Try to extract Ollama's error message from the response body
        try:
            body = resp.json()
            msg = body.get("error", "")
        except Exception:
            msg = resp.text[:200]
        if "model" in msg.lower() and ("not found" in msg.lower() or "pull" in msg.lower()):
            raise RuntimeError(
                f"Model '{model}' not found on Ollama server.\n"
                f"Run: ollama pull {model}\n"
                f"Or check available models with: ollama list"
            )
        raise RuntimeError(
            f"Ollama server error (500) for model '{model}'.\n"
            f"Details: {msg or 'no details — check Ollama server logs'}\n"
            f"Common causes: model not pulled, out of VRAM, or invalid model name."
        )
    resp.raise_for_status()


def _stream_chat(url: str, payload: dict) -> Iterator[dict]:
    with _client.stream("POST", url, json=payload, timeout=300) as resp:
        _raise_friendly(resp, payload["model"])
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def check_connection() -> bool:
    try:
        resp = _client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    resp = _client.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
