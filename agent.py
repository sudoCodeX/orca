"""
Core agent loop: sends messages to Ollama, handles tool calls, repeats.

Pipeline per user turn:
  1. THINK  — model reasons about the task, produces a plan (no tools)
  2. ACT    — model executes the plan using tools (loop until done)

Tool-call format handling:
  1. Native Ollama: message.tool_calls = [{function: {name, arguments}}]
  2. JSON-in-content fallback: content = '{"name": "...", "arguments": {...}}'
     or NDJSON: one JSON object per line
"""

import json
import re
import time
import concurrent.futures
from dataclasses import dataclass
from typing import Callable

import ollama_client
from tools import TOOL_DEFINITIONS, execute_tool
import config
from config import get_system_prompt, get_base_prompt

_TOOL_NAMES = {t["function"]["name"] for t in TOOL_DEFINITIONS}

# ── Speculative THINK cache ───────────────────────────────────────────────────
_think_cache: dict[str, str] = {}


def _should_skip_think(message: str, cache_key: str) -> str | None:
    """Return a cached plan if the message is short and the cached plan is workspace-mode."""
    if len(message.split()) > 20:
        return None
    plan = _think_cache.get(cache_key)
    if plan is None:
        return None
    if "MODE: workspace" in plan:
        return plan
    return None


_THINK_PROMPT = """\
Output a JSON object choosing how to answer. Use exactly these fields:

{
  "mode":      "workspace" | "web" | "direct",
  "what":      "<one sentence — what the user wants>",
  "files":     "<paths/globs to check, or null>",
  "web_query": "<search query, or null>",
  "steps":     "<ordered steps, e.g. web_search → web_fetch → answer>"
}

mode rules:
- "workspace" — ALWAYS use this when the user mentions a file, error, bug, log, or anything
                in the current project. Also use for: find/search/fix/look/check/debug/
                grep/read + anything in the codebase. When in doubt, use workspace.
- "web"       — needs current/live information: weather, news, prices, recent events,
                or documentation for an external library not in the workspace.
- "direct"    — pure coding help, math, explanations with NO reference to local files.
                Only use when the request is entirely self-contained and generic.

Output ONLY the JSON object. No prose, no markdown fences."""

# JSON schema enforced via Ollama format parameter (forces compact structured output)
_THINK_FORMAT = {
    "type": "object",
    "properties": {
        "mode":      {"type": "string", "enum": ["workspace", "web", "direct"]},
        "what":      {"type": "string"},
        "files":     {"type": ["string", "null"]},
        "web_query": {"type": ["string", "null"]},
        "steps":     {"type": "string"},
    },
    "required": ["mode", "what", "files", "web_query", "steps"],
}

_ACT_PROMPT = """\
Execute the plan. Follow the mode strictly:

- mode "direct":    Answer entirely from your own knowledge. Do NOT call any tools.
- mode "workspace": Use search_files / grep_content to locate files, read_file to read \
  them, then synthesise a thorough answer quoting actual file contents.
- mode "web":       Call web_search, then ALWAYS call web_fetch on the most relevant \
  result URL to get the full page content. Never summarise from snippets alone — \
  you must fetch and read the actual page. Then give a specific, factual answer.

IMPORTANT — avoid redundant tool calls:
- The system prompt already contains outlines for every source file in the workspace.
  Do NOT call read_file_outline on files that are already indexed above — use read_symbol directly.
- Do NOT call list_directory on '.' or './' — the workspace tree is already in your system prompt.
- After seeing an outline, use read_symbol(path, name) for the specific function you need — NEVER call read_file on the whole file.
- read_file without start_line/end_line on an already-outlined file will be BLOCKED.

Be thorough and specific. Give the user a complete, useful response."""


# ── Structured JSONL logging ──────────────────────────────────────────────────
# Written to ~/.lca_session.jsonl — one JSON object per LLM call.
# Each entry records: specialist, step, model, tokens, iterations, tools called.
# Disabled if the env var LCA_LOG=0 is set.

import os as _os
import json as _json_log

_LOG_ENABLED = _os.environ.get("LCA_LOG", "1") != "0"
_LOG_PATH = _os.path.expanduser("~/.lca_session.jsonl")


def _log_entry(entry: dict) -> None:
    if not _LOG_ENABLED:
        return
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json_log.dumps(entry) + "\n")
    except OSError:
        pass


# ── Token tracking ────────────────────────────────────────────────────────────

@dataclass
class StepStats:
    """Token and timing stats for one LLM call."""
    step: str           # "think" | "act:N"
    prompt_tokens: int  # tokens in the context window (prompt_eval_count)
    gen_tokens: int     # tokens generated (eval_count)
    duration_ms: int    # total call duration in ms
    tokens_per_sec: float
    prefill_ms: int = 0 # time to process the prompt (prompt_eval_duration)


@dataclass
class SessionStats:
    """Accumulated token totals for the whole session."""
    total_prompt_tokens: int = 0
    total_gen_tokens: int = 0
    total_calls: int = 0

    def add(self, step: StepStats):
        self.total_prompt_tokens += step.prompt_tokens
        self.total_gen_tokens += step.gen_tokens
        self.total_calls += 1


def _extract_stats(step: str, raw: dict) -> StepStats:
    prompt_tokens  = raw.get("prompt_eval_count", 0)
    gen_tokens     = raw.get("eval_count", 0)
    duration_ns    = raw.get("total_duration", 0)
    eval_ns        = raw.get("eval_duration", 0)
    prefill_ns     = raw.get("prompt_eval_duration", 0)
    duration_ms    = duration_ns // 1_000_000
    prefill_ms     = prefill_ns // 1_000_000
    tps = (gen_tokens / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    return StepStats(
        step=step,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        duration_ms=duration_ms,
        tokens_per_sec=tps,
        prefill_ms=prefill_ms,
    )


# ── Tool-call parsing helpers ─────────────────────────────────────────────────

def _try_parse_single(item: dict) -> dict | None:
    if "name" in item and item.get("name") in _TOOL_NAMES:
        return {
            "function": {
                "name": item["name"],
                "arguments": item.get("arguments", item.get("parameters", {})),
            }
        }
    if "tool" in item and item.get("tool") in _TOOL_NAMES:
        return {
            "function": {
                "name": item["tool"],
                "arguments": item.get("parameters", item.get("arguments", {})),
            }
        }
    return None


def _validate_native_tool_calls(tool_calls: list) -> list:
    """Validate native tool_calls entries — some models emit malformed structs.

    A valid entry must have: function.name (non-empty str in _TOOL_NAMES) and
    function.arguments (dict or JSON string). Returns only the valid entries,
    or an empty list if all are malformed (triggers content fallback).
    """
    valid = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments")
        if not isinstance(name, str) or name not in _TOOL_NAMES:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        valid.append({"function": {"name": name, "arguments": args}})
    return valid


_BROKEN_TOOL_TOKENS = re.compile(
    r"<\|tool_(?:response|call)\|?>|<\|function_calls\|>|call:[a-z_]+\{",
    re.I,
)


def _detect_broken_tool_format(content: str) -> bool:
    """Return True if the model is leaking raw internal tool-call tokens.

    This happens when a model (often a custom GGUF) doesn't have a proper
    Ollama-compatible chat template — it outputs Gemma / Llama / Mistral
    special tokens as plain text instead of structured JSON tool calls.
    """
    return bool(_BROKEN_TOOL_TOKENS.search(content))


def _parse_content_tool_calls(content: str) -> list[dict] | None:
    if not content:
        return None
    stripped = content.strip()
    if not stripped.startswith(("{", "[")):
        return None

    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)

    try:
        data = json.loads(stripped)
        items = data if isinstance(data, list) else [data]
        calls = [c for item in items if isinstance(item, dict) for c in [_try_parse_single(item)] if c]
        if calls:
            return calls
    except json.JSONDecodeError:
        pass

    calls = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                call = _try_parse_single(obj)
                if call:
                    calls.append(call)
                else:
                    return None
        except json.JSONDecodeError:
            return None

    return calls if calls else None


# ── Think step ────────────────────────────────────────────────────────────────

def _think(
    user_message: str,
    history: list,
    session: SessionStats,
    on_think: Callable[[str, StepStats], None] | None = None,
    on_llm_start: Callable[[str], None] | None = None,
    on_llm_end: Callable[[], None] | None = None,
    system_prompt: str | None = None,
    model: str | None = None,
    cache_key: str | None = None,
) -> str:
    think_messages = (
        [{"role": "system", "content": system_prompt or get_system_prompt()}]
        + history
        + [
            {"role": "user", "content": user_message},
            {"role": "user", "content": _THINK_PROMPT},
        ]
    )
    if on_llm_start:
        on_llm_start("think")
    raw = ollama_client.chat(
        messages=think_messages, tools=[], stream=False, fmt=_THINK_FORMAT, model=model,
    )
    if on_llm_end:
        on_llm_end()

    content = raw.get("message", {}).get("content", "").strip()

    # Parse the JSON plan; fall back to raw text if the model ignores format
    try:
        plan_data = json.loads(content)
        plan = (
            f"MODE: {plan_data.get('mode', 'direct')}\n"
            f"WHAT: {plan_data.get('what', '')}\n"
            f"FILES: {plan_data.get('files') or 'none'}\n"
            f"WEB_QUERY: {plan_data.get('web_query') or 'none'}\n"
            f"STEPS: {plan_data.get('steps', '')}"
        )
    except (json.JSONDecodeError, AttributeError):
        plan = content  # model didn't respect format, use raw

    stats = _extract_stats("think", raw)
    session.add(stats)
    if cache_key is not None:
        _think_cache[cache_key] = plan
    if on_think:
        on_think(plan, stats)
    return plan


# ── Helpers ───────────────────────────────────────────────────────────────────

# ── Permission system ─────────────────────────────────────────────────────────

# Tools that mutate the filesystem or execute shell commands require user approval.
TOOLS_REQUIRING_PERMISSION = {
    "write_file", "edit_file", "replace_lines", "create_file", "delete_file", "run_command"
}


def describe_tool_action(name: str, arguments: dict) -> str:
    """
    Return a plain-English description of what a tool is about to do.
    Used in permission prompts so the user knows exactly what will happen.
    """
    from config import WORKING_DIR
    from pathlib import Path

    def resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else Path(WORKING_DIR) / path)

    if name == "create_file":
        path = resolve(arguments.get("path", "?"))
        lines = len((arguments.get("content") or "").splitlines())
        return f"Create new file  {path}  ({lines} lines)"

    if name == "write_file":
        path = resolve(arguments.get("path", "?"))
        lines = len((arguments.get("content") or "").splitlines())
        return f"Overwrite file   {path}  ({lines} lines)"

    if name == "replace_lines":
        path = resolve(arguments.get("path", "?"))
        s = arguments.get("start_line", "?")
        e = arguments.get("end_line", "?")
        new = (arguments.get("new_content") or "").strip()[:120]
        return (
            f"Replace lines    {path}  (lines {s}–{e})\n"
            f"  + new: {new!r}"
        )

    if name == "edit_file":
        path = resolve(arguments.get("path", "?"))
        old = (arguments.get("old_string") or "").strip()[:120]
        new = (arguments.get("new_string") or "").strip()[:120]
        return (
            f"Edit file        {path}\n"
            f"  - remove: {old!r}\n"
            f"  + insert: {new!r}"
        )

    if name == "delete_file":
        path = resolve(arguments.get("path", "?"))
        return f"DELETE file      {path}"

    if name == "run_command":
        cmd = arguments.get("command", "?")
        cwd = arguments.get("working_dir", WORKING_DIR)
        return f"Run command      $ {cmd}\n  in: {cwd}"

    # Fallback
    args_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in arguments.items())
    return f"{name}({args_str})"


# ── Tool result compression ───────────────────────────────────────────────────
# Long tool results (file reads, grep output) are passed verbatim on the first
# LLM call after the tool runs — the model needs the full content.
# On subsequent iterations those messages stay in the context window and cost
# tokens without adding value.  We compress them to a short summary using the
# fast model so context stays lean.

_TOOL_COMPRESS_THRESHOLD_BASE = 800  # floor (chars) — used at small context sizes
_TOOL_COMPRESS_PROMPT = (
    "Summarise this tool result in 1–4 bullet points. "
    "Keep: file paths, function names, line numbers, key facts, error messages. "
    "Drop: repeated boilerplate, line-number prefixes, filler text. "
    "Output ONLY the bullets."
)
_COMPRESSED_TAG = "[summarised] "


def _tool_compress_threshold() -> int:
    """Compression threshold scales with the context window.

    At 32k: ~800 chars.  At 128k: ~3200 chars.  At 8k: 800 (floor).
    Keeps proportionally more tool content when the window is large.
    """
    return max(_TOOL_COMPRESS_THRESHOLD_BASE, config.NUM_CTX // 40)


def _compress_tool_message(content: str) -> str:
    """Summarise a long tool result using the fast model.
    Falls back to a truncated version if the LLM call fails.
    """
    threshold = _tool_compress_threshold()
    if content.startswith(_COMPRESSED_TAG):
        return content  # already compressed
    if len(content) <= threshold:
        return content  # short enough, keep as-is

    try:
        raw = ollama_client.chat(
            messages=[
                {"role": "user", "content": f"{_TOOL_COMPRESS_PROMPT}\n\n{content}"},
            ],
            tools=[],
            stream=False,
            model=config.MODEL,
        )
        summary = raw.get("message", {}).get("content", "").strip()
        if summary:
            return _COMPRESSED_TAG + summary
    except Exception:
        pass

    # Fallback: hard truncate with a note
    threshold = _tool_compress_threshold()
    return content[:threshold] + f"\n… [truncated {len(content)} chars]"


_BATCH_COMPRESS_FORMAT = {
    "type": "object",
    "properties": {"summaries": {"type": "array", "items": {"type": "string"}}},
    "required": ["summaries"],
}
_BATCH_COMPRESS_PROMPT = (
    "Summarise each of these tool results. "
    "Return a JSON object with a 'summaries' array — one string per input, same order. "
    "Keep: file paths, function names, line numbers, key facts, error messages. "
    "Drop: repeated boilerplate, line-number prefixes, filler text."
)


def _compress_stale_tool_results(messages: list) -> None:
    """In-place: compress all role=tool messages except the most recent one.

    When multiple results need compression they are batched into a single LLM
    call instead of N sequential calls.  Falls back to individual compression
    if the batch call fails or returns a mismatched count.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # Leave the most recent tool result untouched — model may still reference it
    to_compress = [
        (i, messages[i]["content"])
        for i in tool_indices[:-1]
        if len(messages[i]["content"]) > _tool_compress_threshold()
        and not messages[i]["content"].startswith(_COMPRESSED_TAG)
    ]
    if not to_compress:
        return

    if len(to_compress) == 1:
        i, content = to_compress[0]
        messages[i] = {"role": "tool", "content": _compress_tool_message(content)}
        return

    # Batch: one LLM call for all stale results
    batch_input = json.dumps([c for _, c in to_compress])
    try:
        raw = ollama_client.chat(
            messages=[
                {"role": "user", "content": f"{_BATCH_COMPRESS_PROMPT}\n\n{batch_input}"},
            ],
            tools=[],
            stream=False,
            fmt=_BATCH_COMPRESS_FORMAT,
            model=config.MODEL,
        )
        result_text = raw.get("message", {}).get("content", "").strip()
        batch_out = json.loads(result_text)
        summaries = batch_out.get("summaries", [])
        if isinstance(summaries, list) and len(summaries) == len(to_compress):
            for (i, _), summary in zip(to_compress, summaries):
                if summary:
                    messages[i] = {"role": "tool", "content": _COMPRESSED_TAG + summary}
            return
    except Exception:
        pass

    # Fallback: compress individually
    for i, content in to_compress:
        messages[i] = {"role": "tool", "content": _compress_tool_message(content)}


# ── Code-in-content detector (Fix 1) ─────────────────────────────────────────
# Detects when the model prints a fenced code block with a filename comment
# instead of calling write_file/edit_file, and returns a correction message.

_CODE_FENCE_RE = re.compile(
    r"```(?:python|js|javascript|ts|typescript|go|rust|java|c|cpp|sh|bash|yaml|json|toml)?"
    r"\s*(?:#|//|<!--)?\s*([\w/.\-]+\.\w+)",  # optional filename in comment
    re.I,
)
_FILE_MUTATING_TOOLS = {"write_file", "edit_file", "create_file"}


def _detect_code_in_content(content: str, tools_called: set[str]) -> str | None:
    """Return a correction prompt if the model printed code instead of calling a tool.

    Only fires when:
    - Content contains a fenced code block (possibly with a filename)
    - AND no file-mutating tool was called this iteration
    Returns None if no intervention is needed.
    """
    if not content:
        return None
    if tools_called & _FILE_MUTATING_TOOLS:
        return None  # model did call a file tool, no problem
    if "```" not in content:
        return None
    if not _CODE_FENCE_RE.search(content):
        # Has code blocks but no filename hint — only flag if fairly long
        if content.count("```") < 2 or len(content) < 200:
            return None
    return (
        "You printed code in your response instead of calling a file tool. "
        "The file on disk has NOT changed. "
        "Call edit_file (for changes to an existing file) or write_file (for a new file) NOW "
        "to apply the changes. Do not print the code again — just call the tool."
    )


# ── Streaming collector ───────────────────────────────────────────────────────

def _stream_collect(
    messages: list,
    tools: list,
    model: str | None,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    """Stream a chat call and return a reconstructed raw dict.

    Calls on_token for each content chunk UNLESS the first chunk looks like
    tool-call JSON (starts with '{' or '['), in which case content is buffered
    silently.  This prevents raw JSON from leaking to the terminal when the
    model is composing a tool call in its content field.
    """
    content_parts: list[str] = []
    tool_calls: list = []
    final_chunk: dict = {}
    streaming_to_terminal: bool | None = None  # None = undecided
    broken_format_detected: bool = False

    for chunk in ollama_client.chat(messages=messages, tools=tools, stream=True, model=model):
        msg = chunk.get("message", {})
        piece = msg.get("content") or ""

        if piece:
            # Decide on first non-empty chunk whether to stream to terminal.
            # Also check for broken tool-call token formats on the first chunk —
            # if detected, suppress all streaming immediately so garbage doesn't
            # reach the terminal at all.
            if streaming_to_terminal is None:
                first_char = piece.lstrip()[:1]
                streaming_to_terminal = first_char not in ("{", "[")
                if streaming_to_terminal and _detect_broken_tool_format(piece):
                    streaming_to_terminal = False
                    broken_format_detected = True

            content_parts.append(piece)
            # Re-check accumulated buffer every few chunks in case the broken
            # tokens span multiple pieces (model may split <|tool_response> across chunks).
            if streaming_to_terminal and not broken_format_detected:
                accumulated = "".join(content_parts)
                if _detect_broken_tool_format(accumulated):
                    streaming_to_terminal = False
                    broken_format_detected = True
            if on_token and streaming_to_terminal:
                on_token(piece)

        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]

        if chunk.get("done"):
            final_chunk = chunk

    return {
        "message": {
            "role": "assistant",
            "content": "".join(content_parts),
            **( {"tool_calls": tool_calls} if tool_calls else {} ),
        },
        "prompt_eval_count": final_chunk.get("prompt_eval_count", 0),
        "eval_count":        final_chunk.get("eval_count", 0),
        "total_duration":    final_chunk.get("total_duration", 0),
        "eval_duration":     final_chunk.get("eval_duration", 0),
    }


# ── Act loop ──────────────────────────────────────────────────────────────────

# Timeout (seconds) waiting for user permission. 0 = no timeout (wait forever).
PERMISSION_TIMEOUT: float = float(_os.environ.get("LCA_PERMISSION_TIMEOUT", "0"))


def _run_tool_with_permission(
    name: str,
    arguments: dict,
    on_tool_call: Callable | None,
    on_tool_result: Callable | None,
    on_permission_request: Callable | None,
) -> str | None:
    """Execute one tool, handling permission check.

    Returns the result string to append to messages.  Denial and timeout cases
    return an informational string rather than raising so the model can react.
    """
    if on_tool_call:
        on_tool_call(name, arguments)

    if name in TOOLS_REQUIRING_PERMISSION and on_permission_request:
        description = describe_tool_action(name, arguments)
        if PERMISSION_TIMEOUT > 0:
            import threading
            _perm_result: list[bool] = []
            def _ask():
                _perm_result.append(on_permission_request(name, arguments, description))
            t = threading.Thread(target=_ask, daemon=True)
            t.start()
            t.join(timeout=PERMISSION_TIMEOUT)
            if not _perm_result:
                result = f"Permission timed out after {PERMISSION_TIMEOUT:.0f}s: {name} was skipped."
                if on_tool_result:
                    on_tool_result(name, result)
                return result
            approved = _perm_result[0]
        else:
            approved = on_permission_request(name, arguments, description)

        if not approved:
            result = f"User denied permission: {name} was not executed."
            if on_tool_result:
                on_tool_result(name, result)
            return result

    result = execute_tool(name, arguments)
    if on_tool_result:
        on_tool_result(name, result)
    return result


def _act(
    user_message: str,
    plan: str,
    history: list,
    session: SessionStats,
    on_tool_call: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str], None] | None = None,
    on_act_stats: Callable[[StepStats], None] | None = None,
    on_llm_start: Callable[[str], None] | None = None,
    on_llm_end: Callable[[], None] | None = None,
    on_permission_request: Callable[[str, dict, str], bool] | None = None,
    on_stream_token: Callable[[str], None] | None = None,
    max_iterations: int = 20,
    system_prompt: str | None = None,
    tool_defs: list | None = None,
    enable_compression: bool = True,
    model: str | None = None,
    specialist: str = "",
) -> str:
    messages = (
        [{"role": "system", "content": system_prompt or get_system_prompt()}]
        + history
        + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"[My plan]\n{plan}"},
            {"role": "user", "content": _ACT_PROMPT},
        ]
    )

    _log_tools_called: list[str] = []
    _act_start = time.monotonic()
    active_tools = tool_defs or TOOL_DEFINITIONS

    # Fix A: per-turn dedup cache for read-only tools.
    # If the model calls the same read_file / read_file_outline / etc. twice in one
    # turn, return a short cache-hit note instead of re-running the tool and
    # bloating the context with duplicate content.
    _READ_ONLY_TOOLS = {
        "read_file", "read_file_outline", "read_symbol",
        "list_directory", "search_files", "grep_content",
    }
    _tool_result_cache: dict[str, str] = {}

    def _cache_key(name: str, arguments: dict) -> str:
        return f"{name}:{json.dumps(arguments, sort_keys=True)}"

    def _execute_cached(name: str, arguments: dict) -> str:
        if name in _READ_ONLY_TOOLS:
            key = _cache_key(name, arguments)
            if key in _tool_result_cache:
                cached = _tool_result_cache[key]
                # For outline hits: extract symbol names so the model can jump
                # straight to read_symbol without needing another LLM call to digest
                # the full outline result.
                if name == "read_file_outline":
                    path = arguments.get("path", "")
                    names = re.findall(r"L\d+\s+(\w+)", cached)[:20]
                    symbols = ", ".join(names) if names else "see system prompt above"
                    return (
                        f"[already indexed in system prompt — do NOT call this again. "
                        f"Call read_symbol(path='{path}', symbol='<name>') directly. "
                        f"Symbols in this file: {symbols}]"
                    )
                return "[cached — already in context above]"
            result = execute_tool(name, arguments)
            _tool_result_cache[key] = result
            return result
        return execute_tool(name, arguments)

    # Pre-seed the dedup cache with outlines already in the system prompt code index.
    # This makes any read_file_outline call on an indexed file return the cache-hit
    # note immediately — skipping a 15-30s LLM exploration iteration entirely.
    try:
        from pathlib import Path as _SeedPath
        from tools.file_ops import read_file_outline as _rfo
        _CODE_EXTS = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
            ".go", ".rs", ".java", ".kt", ".rb",
            ".c", ".cpp", ".h", ".hpp",
            ".json", ".yaml", ".yml", ".toml",
        }
        for _f in sorted(_SeedPath(config.WORKING_DIR).iterdir()):
            if _f.is_file() and not _f.name.startswith(".") and _f.suffix.lower() in _CODE_EXTS:
                _rel = str(_f.relative_to(config.WORKING_DIR))
                _outline = _rfo(_rel)
                # Seed both relative and absolute path variants
                _tool_result_cache[_cache_key("read_file_outline", {"path": _rel})] = _outline
                _tool_result_cache[_cache_key("read_file_outline", {"path": str(_f)})] = _outline
    except Exception:
        pass

    # Fix C: track which files have been outlined this turn so we can warn
    # when the model then requests the full file without a line range.
    _outlined_files: set[str] = set()

    # Loop-guard state.
    # _recent_calls: sliding window of the last 3 (name, args_key) tuples.
    # _stall_streak: consecutive iterations where every tool result was
    #   cached/blocked (no real work done). After 3 stalls inject a nudge;
    #   after 5 abort to avoid wasting tokens.
    _recent_calls: list[str] = []
    _stall_streak: int = 0
    _STALL_WARN  = 3
    _STALL_ABORT = 5

    for iteration in range(max_iterations):
        if on_llm_start:
            on_llm_start("act")
        # Stream all ACT calls: content tokens flow to terminal for text responses;
        # tool-call JSON is buffered silently (first char heuristic in _stream_collect).
        raw = _stream_collect(messages=messages, tools=active_tools, model=model, on_token=on_stream_token)
        if on_llm_end:
            on_llm_end()
        message = raw.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls") or []

        # Detect models that leak raw internal tool-call tokens (broken chat template).
        # Abort immediately with a clear message rather than looping 20 times.
        if _detect_broken_tool_format(content):
            return (
                f"⚠ Model '{model or config.MODEL}' is outputting raw internal tool-call tokens "
                f"(e.g. <|tool_response|>, call:func{{...}}) instead of proper JSON tool calls.\n"
                f"This means the model's chat template does not support Ollama tool calling.\n"
                f"Switch to a supported model: gemma4:e4b, qwen2.5-coder:7b, or gemma4:27b."
            )

        # Validate native tool_calls — fall back to content parsing if malformed
        tool_calls = _validate_native_tool_calls(raw_tool_calls)
        if raw_tool_calls and not tool_calls:
            parsed = _parse_content_tool_calls(content)
            if parsed:
                tool_calls = parsed
                content = ""
        elif not tool_calls:
            parsed = _parse_content_tool_calls(content)
            if parsed:
                tool_calls = parsed
                content = ""

        stats = _extract_stats(f"act:{iteration + 1}", raw)
        session.add(stats)
        if on_act_stats:
            on_act_stats(stats)

        tools_called_this_iter: set[str] = {
            c.get("function", {}).get("name", "") for c in tool_calls
        }

        assistant_msg: dict = {"role": role, "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            correction = _detect_code_in_content(content, set(_log_tools_called))
            if correction:
                messages.append({"role": "user", "content": correction})
                continue

            _log_entry({
                "ts": time.time(),
                "specialist": specialist,
                "model": model or "",
                "iterations": iteration + 1,
                "tools_called": _log_tools_called,
                "prompt_tokens": stats.prompt_tokens,
                "gen_tokens": stats.gen_tokens,
                "duration_ms": int((time.monotonic() - _act_start) * 1000),
            })
            return content

        _log_tools_called.extend(tools_called_this_iter)

        if enable_compression:
            _compress_stale_tool_results(messages)

        # Normalise arguments for all calls upfront
        calls_norm: list[tuple[str, dict]] = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls_norm.append((name, arguments))

        # Separate permission-requiring calls (must be sequential + interactive)
        # from free calls (can run in parallel).
        perm_indices = [
            i for i, (name, _) in enumerate(calls_norm)
            if name in TOOLS_REQUIRING_PERMISSION and on_permission_request
        ]
        free_indices = [i for i in range(len(calls_norm)) if i not in set(perm_indices)]

        tool_results: dict[int, str] = {}

        # Sequential: permission-gated tools
        for i in perm_indices:
            name, arguments = calls_norm[i]
            # Permission tools go through _run_tool_with_permission which calls
            # execute_tool internally; dedup cache is not applied to mutating tools.
            result = _run_tool_with_permission(name, arguments, on_tool_call, on_tool_result, on_permission_request)
            tool_results[i] = result

        # Parallel: free tools (announce first so display stays coherent)
        # NOTE: on_tool_result is NOT called here — it is called after Fix C below
        # so the display always shows the final result (possibly the BLOCKED message)
        # rather than the raw content before interception.
        if len(free_indices) > 1:
            for i in free_indices:
                name, arguments = calls_norm[i]
                if on_tool_call:
                    on_tool_call(name, arguments)

            def _run_free(i: int) -> tuple[int, str]:
                n, a = calls_norm[i]
                return i, _execute_cached(n, a)  # Fix A: dedup cache

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(free_indices), 4)) as ex:
                for idx, res in ex.map(_run_free, free_indices):
                    tool_results[idx] = res
        else:
            for i in free_indices:
                name, arguments = calls_norm[i]
                if on_tool_call:
                    on_tool_call(name, arguments)
                tool_results[i] = _execute_cached(name, arguments)  # Fix A: dedup cache

        # Fix C: track outlined files; intercept full read_file after outline.
        # Paths are normalised to absolute so relative vs absolute calls both match.
        for i, (name, arguments) in enumerate(calls_norm):
            res = tool_results.get(i, "")
            if name == "read_file_outline":
                raw_path = arguments.get("path", "")
                if raw_path:
                    try:
                        from pathlib import Path as _Path
                        abs_path = str((_Path(config.WORKING_DIR) / raw_path).resolve()
                                       if not _Path(raw_path).is_absolute()
                                       else _Path(raw_path).resolve())
                        # Store (abs_path → outline_result) so we can quote symbols later
                        _outlined_files.add(abs_path)
                        _outlined_files.add(raw_path)  # also store raw in case match is exact
                    except Exception:
                        _outlined_files.add(raw_path)
            elif name == "read_file":
                raw_path = arguments.get("path", "")
                try:
                    from pathlib import Path as _Path
                    abs_path = str((_Path(config.WORKING_DIR) / raw_path).resolve()
                                   if not _Path(raw_path).is_absolute()
                                   else _Path(raw_path).resolve())
                except Exception:
                    abs_path = raw_path
                already_outlined = abs_path in _outlined_files or raw_path in _outlined_files
                if (already_outlined
                        and "start_line" not in arguments
                        and "end_line" not in arguments
                        and not res.startswith("[cached")):
                    # Hard intercept: replace full content with a redirect message.
                    # The outline is already in context — force the model to use read_symbol.
                    tool_results[i] = (
                        "[BLOCKED: you already have the outline for this file in your context above. "
                        "Reading the full file again wastes context. "
                        "Use read_symbol(path, symbol_name) for the specific function/class you need, "
                        "or read_file with start_line/end_line if you have line numbers from the outline. "
                        "Do NOT call read_file on this path without a line range.]"
                    )

        # ── Loop-guard ────────────────────────────────────────────────────────
        # 1. Consecutive-repeat detection: if the model calls the exact same tool
        #    with the same args 3 times in a row, it is stuck. Inject an abort.
        # 2. Stall detection: if every result this iteration was cached/blocked
        #    (no real work), increment the stall streak. After _STALL_WARN warn;
        #    after _STALL_ABORT stop entirely.
        _NON_WORK_PREFIXES = ("[cached", "[already indexed", "[BLOCKED")
        iter_call_keys = [
            _cache_key(name, arguments) for name, arguments in calls_norm
        ]
        _recent_calls.extend(iter_call_keys)
        _recent_calls = _recent_calls[-9:]  # keep last 9 entries (3 iterations × 3 calls)

        # Check for the same single call repeated 3 times back-to-back
        if len(iter_call_keys) == 1:
            key = iter_call_keys[0]
            if _recent_calls.count(key) >= 3:
                messages.append({
                    "role": "tool",
                    "content": (
                        f"[LOOP DETECTED: you have called '{calls_norm[0][0]}' with the same "
                        f"arguments {_recent_calls.count(key)} times in a row. Stop. "
                        "You are stuck. Either you have all the information you need — "
                        "proceed to make the edit — or explain what is blocking you."
                    ),
                })
                break

        all_stalled = all(
            tool_results.get(i, "").startswith(_NON_WORK_PREFIXES)
            for i in range(len(calls_norm))
        )
        if all_stalled:
            _stall_streak += 1
        else:
            _stall_streak = 0

        if _stall_streak >= _STALL_ABORT:
            messages.append({
                "role": "user",
                "content": (
                    f"You have made {_stall_streak} consecutive iterations where every tool call "
                    "was blocked or returned a cached result. No progress has been made. "
                    "Stop attempting tool calls. Summarise what you found so far and either "
                    "complete the task with the information you have or tell the user what "
                    "information is missing."
                ),
            })
            break

        if _stall_streak == _STALL_WARN:
            messages.append({
                "role": "user",
                "content": (
                    "Warning: your last 3 iterations have all hit cached or blocked results. "
                    "You already have the information you need in your context. "
                    "Stop reading files and proceed to implement the change."
                ),
            })

        # Notify display and append results in original call order.
        # on_tool_result is called here (after Fix C) so the display reflects the
        # final result — including any BLOCKED intercept messages.
        for i, (name, arguments) in enumerate(calls_norm):
            result = tool_results.get(i, f"Tool {name} was not executed.")
            # Only notify for free tools — permission tools called on_tool_result
            # inside _run_tool_with_permission already.
            if i not in set(perm_indices) and on_tool_result:
                on_tool_result(name, result)

        for i, (name, arguments) in enumerate(calls_norm):
            result = tool_results.get(i, f"Tool {name} was not executed.")
            # Auto-recover from edit_file failures: inject a fresh file read so
            # the model can extract the exact old_string without burning an extra
            # iteration on another read_file call.
            if name == "edit_file" and "old_string not found" in result:
                path = arguments.get("path", "")
                if path:
                    fresh = _execute_cached("read_file", {"path": path})
                    result = (
                        f"{result}\n\n"
                        "[Hint: old_string must match the file exactly. "
                        "Here is the current file content — copy the exact text you want to replace:]\n"
                        f"{fresh}"
                    )
            messages.append({"role": "tool", "content": result})
            if name == "web_search":
                messages.append({
                    "role": "user",
                    "content": (
                        "Review the search results above. "
                        "If the snippets already contain a specific answer to the user's question, "
                        "respond directly now — do NOT call web_fetch. "
                        "Only call web_fetch if you need more detail that isn't in the snippets. "
                        f"User's question: {user_message}"
                    ),
                })

    messages.append({
        "role": "user",
        "content": "Please summarize what you have done so far and give the user a final answer.",
    })
    if on_llm_start:
        on_llm_start("act")
    raw = _stream_collect(messages=messages, tools=[], model=model, on_token=on_stream_token)
    if on_llm_end:
        on_llm_end()
    stats = _extract_stats("act:final", raw)
    session.add(stats)
    if on_act_stats:
        on_act_stats(stats)
    final = raw.get("message", {}).get("content", "")
    _log_entry({
        "ts": time.time(),
        "specialist": specialist,
        "model": model or "",
        "iterations": max_iterations,
        "tools_called": _log_tools_called,
        "prompt_tokens": stats.prompt_tokens,
        "gen_tokens": stats.gen_tokens,
        "duration_ms": int((time.monotonic() - _act_start) * 1000),
        "hit_max_iterations": True,
    })
    return final


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    history: list,
    session: SessionStats | None = None,
    on_think: Callable[[str, StepStats], None] | None = None,
    on_tool_call: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str], None] | None = None,
    on_act_stats: Callable[[StepStats], None] | None = None,
    on_llm_start: Callable[[str], None] | None = None,
    on_llm_end: Callable[[], None] | None = None,
    on_permission_request: Callable[[str, dict, str], bool] | None = None,
    on_stream_token: Callable[[str], None] | None = None,
    max_iterations: int = 20,
) -> tuple[str, list]:
    """
    Run one full THINK → ACT cycle for a user message.

    Args:
        user_message:   The user's raw input.
        history:        Conversation history (updated in-place).
        session:        Shared SessionStats accumulator (created if None).
        on_think:       Callback(plan, StepStats) after the think step.
        on_tool_call:   Callback(name, arguments) before each tool runs.
        on_tool_result: Callback(name, result) after each tool runs.
        on_act_stats:   Callback(StepStats) after each act LLM call.
        max_iterations: Max tool-call rounds in the ACT phase.

    Returns:
        (final_answer, updated_history)
    """
    if session is None:
        session = SessionStats()

    history.append({"role": "user", "content": user_message})

    plan = _think(
        user_message, history[:-1], session,
        on_think=on_think,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
    )
    # Plan cap: ~3% of context window. Structured JSON plan is ~100 chars;
    # this headroom handles any model that ignores the format constraint.
    from config import NUM_CTX
    plan_cap = max(400, NUM_CTX // 32)
    plan_for_act = plan[:plan_cap] + ("\n…(plan truncated)" if len(plan) > plan_cap else "")

    answer = _act(
        user_message,
        plan_for_act,
        history[:-1],
        session,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_act_stats=on_act_stats,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
        on_permission_request=on_permission_request,
        on_stream_token=on_stream_token,
        max_iterations=max_iterations,
    )

    history.append({"role": "assistant", "content": answer})
    return answer, history


def make_history() -> list:
    return []


# ── Rolling history compression ───────────────────────────────────────────────
# Keep the last HISTORY_KEEP turns verbatim; summarise everything older into
# a single assistant message so the context window stays bounded.
# Threshold: compress when history exceeds HISTORY_COMPRESS_AFTER turns.
# One "turn" = one user message + one assistant reply = 2 messages.

def _history_thresholds() -> tuple[int, int]:
    """Return (compress_after, keep) turn counts scaled to the context window.

    At 32k:  compress_after=10, keep=6  (original hardcoded values)
    At 128k: compress_after=43, keep=21
    At 8k:   compress_after=10, keep=6  (floors hold)
    """
    compress_after = max(10, config.NUM_CTX // 3000)
    keep = max(6, config.NUM_CTX // 6000)
    return compress_after, keep

_COMPRESS_PROMPT = """\
Summarise the conversation above in 3–6 bullet points.
Focus on: decisions made, files created/edited, key facts established.
Be specific — include filenames, function names, and concrete outcomes.
Output ONLY the bullet list, no preamble."""


def maybe_compress_history(
    history: list,
    on_llm_start: Callable | None = None,
    on_llm_end: Callable | None = None,
) -> list:
    """If history is long, compress old turns into a summary and return the
    trimmed list.  Returns history unchanged if compression is not needed."""
    compress_after, keep = _history_thresholds()
    turns = len(history) // 2  # each turn = user + assistant message
    if turns <= compress_after:
        return history

    # Split: everything except the last `keep` turns gets summarised
    keep_msgs = keep * 2
    to_summarise = history[:-keep_msgs]
    to_keep = history[-keep_msgs:]

    summarise_messages = (
        [{"role": "system", "content": get_base_prompt()}]
        + to_summarise
        + [{"role": "user", "content": _COMPRESS_PROMPT}]
    )
    if on_llm_start:
        on_llm_start("compress")
    raw = ollama_client.chat(messages=summarise_messages, tools=[], stream=False, model=config.MODEL)
    if on_llm_end:
        on_llm_end()

    summary = raw.get("message", {}).get("content", "").strip()

    # Fix 5: extract a deduplicated index of files read/written in compressed turns.
    # This preserves tool-call structure as structured metadata rather than prose,
    # so the model can answer "what files did I already read?" from compressed history.
    files_read: list[str] = []
    files_written: list[str] = []
    _path_re = re.compile(r'path="([^"]+)"')
    for msg in to_summarise:
        role_m = msg.get("role", "")
        content_m = msg.get("content", "") or ""
        if role_m == "tool":
            # tool results from read_file contain path="..." in the XML wrapper
            for p in _path_re.findall(content_m):
                if p not in files_read:
                    files_read.append(p)
        if role_m == "assistant":
            # look for write_file/edit_file/create_file in tool_calls
            for tc in msg.get("tool_calls", []):
                tc_name = tc.get("function", {}).get("name", "")
                if tc_name in {"write_file", "edit_file", "create_file"}:
                    p = tc.get("function", {}).get("arguments", {}).get("path", "")
                    if p and p not in files_written:
                        files_written.append(p)

    index_parts = []
    if files_read:
        index_parts.append("Files read: " + ", ".join(files_read))
    if files_written:
        index_parts.append("Files written/edited: " + ", ".join(files_written))
    index_block = ("\n[File access index]\n" + "\n".join(index_parts)) if index_parts else ""

    summary_msg = {
        "role": "assistant",
        "content": f"[Earlier conversation summary]\n{summary}{index_block}",
    }
    return [summary_msg] + to_keep
