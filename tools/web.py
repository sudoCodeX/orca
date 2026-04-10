"""
Web search and fetch tools.

web_search  — DuckDuckGo search (no API key required)
web_fetch   — Fetch URL via Jina Reader (renders JS, returns clean markdown)
              Falls back to trafilatura → markdownify if Jina is unavailable.
"""

import ipaddress
import time
import urllib.parse
import httpx
from tools.cache import web_cache, is_web_stale

# Private/loopback ranges blocked to prevent SSRF
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private_url(url: str) -> bool:
    """Return True if the URL resolves to a private/loopback address."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        # Reject raw IP literals that are private
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _BLOCKED_NETWORKS)
    except ValueError:
        # hostname — allow (DNS resolution happens at request time; we can't
        # block hostnames reliably without resolving, which has its own risks)
        return False

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# Jina Reader: converts any URL to clean markdown (handles JS rendering)
_JINA_BASE = "https://r.jina.ai/"


# ── web_search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 8) -> str:
    """Search DuckDuckGo and return titles, URLs, and full snippets."""
    cache_key = ("search", query, max_results)
    cached, hit = web_cache.get(cache_key)
    if hit:
        cached_at, result = cached
        if not is_web_stale(cached_at, "search"):
            return result

    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs is not installed.\nRun: uv add ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "").strip()
        url     = r.get("href", "").strip()
        snippet = r.get("body", "").strip()
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")   # full snippet — no truncation
        lines.append("")

    result = "\n".join(lines)
    web_cache.put(cache_key, (time.monotonic(), result))
    return result


# ── web_fetch ─────────────────────────────────────────────────────────────────

def web_fetch(url: str, max_chars: int = 0) -> str:
    """
    Fetch a URL and return its main content as clean markdown.

    Primary:  Jina Reader (r.jina.ai) — renders JS, strips boilerplate, returns markdown.
    Fallback: trafilatura → markdownify (pure Python, works offline).

    max_chars defaults to 25% of NUM_CTX (in characters, ~4 chars/token).
    """
    if _is_private_url(url):
        return f"Error: fetching private/loopback addresses is not allowed: {url}"

    cache_key = ("fetch", url, max_chars)
    cached, hit = web_cache.get(cache_key)
    if hit:
        cached_at, result = cached
        if not is_web_stale(cached_at, "fetch"):
            return result

    if max_chars <= 0:
        from config import NUM_CTX
        max_chars = max(6000, NUM_CTX // 4 * 4)   # 25% of ctx window in chars

    markdown = _fetch_via_jina(url) or _fetch_via_trafilatura(url)

    if not markdown:
        return f"Could not extract content from: {url}"

    if len(markdown) > max_chars:
        markdown = markdown[:max_chars] + f"\n\n… (truncated at {max_chars:,} chars)"

    result = f"[{url}]\n\n{markdown}"
    web_cache.put(cache_key, (time.monotonic(), result))
    return result


def _fetch_via_jina(url: str) -> str | None:
    """Use Jina Reader to get clean markdown from any URL (handles JS)."""
    try:
        jina_url = _JINA_BASE + url
        resp = httpx.get(
            jina_url,
            headers={**_HEADERS, "Accept": "text/markdown"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text.strip()
        return text if len(text) > 100 else None
    except Exception:
        return None


def _fetch_via_trafilatura(url: str) -> str | None:
    """Fallback: fetch raw HTML, extract main content, convert to markdown."""
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    # trafilatura extracts the article/main content, removes boilerplate
    try:
        import trafilatura
        cleaned = trafilatura.extract(
            html,
            output_format="xml",
            include_tables=True,
            include_links=False,
            favor_recall=True,
            url=url,
        )
        if cleaned:
            from markdownify import markdownify as md
            return md(cleaned, heading_style="ATX", strip=["a", "img"]).strip()
    except Exception:
        pass

    # Last resort: markdownify on raw HTML stripping noisy elements
    try:
        from markdownify import markdownify as md
        return md(
            html, heading_style="ATX",
            strip=["script", "style", "nav", "header", "footer", "a", "img"],
        ).strip()
    except Exception:
        return None
