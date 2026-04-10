"""
Session-scoped LRU caches for file reads, outlines, and search results.

Invalidation strategy
---------------------
File / outline caches
    Key includes st_mtime_ns (nanosecond mtime from os.stat).
    When a file is written its mtime advances so the old key is never
    looked up again.  Stale entries age out via LRU eviction — no
    explicit invalidation needed.  Works for external edits too (user's
    editor, git checkout, etc.).

Grep / search caches
    These span many files, so per-file mtime keying is impractical.
    Two-layer invalidation:
      1. Dirty clock — any agent write/edit/create/delete advances it;
         entries cached before that moment are considered stale.
      2. TTL (default 30 s) — catches files edited outside the agent.
    Override TTL via LCA_CACHE_SEARCH_TTL (seconds).

Sizing
------
Default max entries:  file/outline → 128,  grep/search → 64
Override via env:     LCA_CACHE_FILE_SIZE, LCA_CACHE_OUTLINE_SIZE,
                      LCA_CACHE_GREP_SIZE,  LCA_CACHE_SEARCH_SIZE
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Accumulates hit/miss/eviction counts for one cache instance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        """Total number of lookups (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that were cache hits."""
        return self.hits / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"hits={self.hits:,}  misses={self.misses:,}  "
            f"rate={self.hit_rate:.0%}  evictions={self.evictions:,}"
        )


# ── LRU cache ─────────────────────────────────────────────────────────────────

class LRUCache:
    """O(1) get/put LRU cache backed by collections.OrderedDict.

    Not thread-safe — designed for the single-threaded agent loop.
    """

    def __init__(self, max_size: int, name: str = "") -> None:
        self._store: OrderedDict = OrderedDict()
        self._max = max_size
        self.name = name
        self.stats = CacheStats()

    def get(self, key: object) -> tuple[object, bool]:
        """Return (value, True) on hit or (None, False) on miss."""
        node = self._store.get(key)
        if node is None:
            self.stats.misses += 1
            return None, False
        self._store.move_to_end(key)
        self.stats.hits += 1
        return node, True

    def put(self, key: object, value: object) -> None:
        """Insert or refresh an entry; evict the LRU entry if over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        self._store[key] = value
        if len(self._store) > self._max:
            self._store.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        """Remove all entries (stats are preserved)."""
        self._store.clear()

    def __len__(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._store)

    def __repr__(self) -> str:
        return f"LRUCache(name={self.name!r}, size={len(self)}/{self._max}, {self.stats})"


# ── Session cache instances ───────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# read_file        key: (resolved_path, mtime_ns, start_line, end_line)
file_cache = LRUCache(max_size=_env_int("LCA_CACHE_FILE_SIZE", 128), name="file")

# read_file_outline  key: (resolved_path, mtime_ns)
outline_cache = LRUCache(max_size=_env_int("LCA_CACHE_OUTLINE_SIZE", 128), name="outline")

# grep_content     key: (pattern, base_path, file_glob, context_lines)
#                  value: (cached_at: float, result: str)
grep_cache = LRUCache(max_size=_env_int("LCA_CACHE_GREP_SIZE", 64), name="grep")

# search_files     key: (pattern, base_path)
#                  value: (cached_at: float, result: str)
search_cache = LRUCache(max_size=_env_int("LCA_CACHE_SEARCH_SIZE", 64), name="search")

# web_search / web_fetch  key: ("search"|"fetch", query_or_url)
#                          value: (cached_at: float, result: str)
# Separate TTLs: search results expire quickly; fetched pages are stable longer.
WEB_SEARCH_TTL: float = float(os.environ.get("LCA_CACHE_WEB_SEARCH_TTL", "60"))
WEB_FETCH_TTL:  float = float(os.environ.get("LCA_CACHE_WEB_FETCH_TTL", "300"))
web_cache = LRUCache(max_size=_env_int("LCA_CACHE_WEB_SIZE", 32), name="web")


def is_web_stale(cached_at: float, kind: str) -> bool:
    """Return True if a web result should be discarded.
    kind: "search" | "fetch"
    """
    ttl = WEB_SEARCH_TTL if kind == "search" else WEB_FETCH_TTL
    return (time.monotonic() - cached_at) > ttl


# ── Workspace dirty clock ─────────────────────────────────────────────────────
# A plain list so we can mutate it from functions without `global`.

_dirty: list[float] = [0.0]  # [0] = monotonic timestamp of last agent write

# TTL for grep/search entries — catches external edits the dirty clock misses.
SEARCH_TTL: float = float(os.environ.get("LCA_CACHE_SEARCH_TTL", "30"))


def mark_workspace_dirty() -> None:
    """Advance the dirty clock after any file mutation."""
    _dirty[0] = time.monotonic()


def is_search_stale(cached_at: float) -> bool:
    """Return True if a search result should be discarded.

    Stale when either:
    - the agent mutated the workspace after the result was cached, or
    - the result is older than SEARCH_TTL seconds (catches external edits).
    """
    return cached_at <= _dirty[0] or (time.monotonic() - cached_at) > SEARCH_TTL


# ── Module-level helpers ──────────────────────────────────────────────────────

def all_stats() -> dict[str, CacheStats]:
    """Return a snapshot of stats for all five caches."""
    return {
        "file":    file_cache.stats,
        "outline": outline_cache.stats,
        "grep":    grep_cache.stats,
        "search":  search_cache.stats,
        "web":     web_cache.stats,
    }


def clear_all() -> None:
    """Wipe all caches and reset the dirty clock (call on session clear)."""
    file_cache.clear()
    outline_cache.clear()
    grep_cache.clear()
    search_cache.clear()
    web_cache.clear()
    _dirty[0] = 0.0
