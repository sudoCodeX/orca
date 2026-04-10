"""
Multi-agent orchestration for lca.

Flow:
  1. Router classifies the user request → picks specialist or a chain
  2. Each specialist runs a full THINK → ACT cycle with its own system prompt + tools
  3. In chain mode each specialist's output is passed as context to the next

Specialists:
  architect  — design, patterns, interfaces, implementation plan. Read-only tools.
  engineer   — writing/editing code, fixing bugs, refactoring. All tools.
  tester     — test writing, coverage analysis, edge cases, QA. All tools.
"""

import json
import re
import threading
from typing import Callable

import ollama_client
from agent import _think, _act, _extract_stats, SessionStats, StepStats, maybe_compress_history, _should_skip_think
from tools import TOOL_DEFINITIONS, execute_tool
from tools.scratchpad import get_context_for_role
import config
from config import get_system_prompt, get_base_prompt


# ── Specialist role prompts ───────────────────────────────────────────────────

_ROLE_PROMPTS: dict[str, str] = {
    "architect": """\
You are a Senior Software Architect embedded in this workspace.

You are only invoked when the user explicitly asks for system design, a new major \
component, or a significant architectural decision — not for everyday feature work.

Responsibilities:
- Survey the existing codebase structure before proposing anything
- Define component boundaries, interfaces, data flow, and file layout
- Identify the best patterns, abstractions, and trade-offs
- Produce a concrete, actionable implementation plan: file paths, \
function/class signatures, and clear reasoning

Constraints:
- Use read_file_outline and grep_content for codebase exploration
- Use web_search and web_fetch to research external APIs, patterns, or documentation
- Do NOT write, edit, or create files — you produce a plan, not implementation
- Be specific: name every file, function, and data structure you recommend
- Keep the plan actionable and scoped to what was asked; do not redesign the entire system
""",

    "engineer": """\
You are a Senior Software Engineer embedded in this workspace.

Responsibilities:
- Implement features, fix bugs, and refactor code
- Follow existing code style, patterns, and conventions
- Make minimal, targeted changes (prefer edit_file over full rewrites)

Rules:
- Always outline or read a file before editing it
- Run the relevant command or tests after changes to verify correctness
- Leave code cleaner than you found it, but only within the scope of the task

CRITICAL — always search the workspace, never reason from memory:
- If the user reports an error, bug, or log entry → grep_content or search_files FIRST
- NEVER assume you know what a file contains — always read or outline it before acting
- "Find the issue" means search the actual files, not explain what the issue might be

CRITICAL — use targeted reads, not full-file reads:
- If you know the function/class name → use read_symbol(path, symbol) to read just that symbol
- read_symbol is faster than read_file_outline → read_file; use it for single-symbol lookups
- Use read_file_outline when you need the full structure; use read_file only as a last resort

CRITICAL — prefer replace_lines over edit_file:
- After read_file_outline you have line numbers → use replace_lines(path, start, end, new_content)
- replace_lines never fails due to whitespace; edit_file requires exact text match
- Only use edit_file when you do NOT have line numbers and the match is unambiguous

CRITICAL — you MUST use tools to make all changes:
- NEVER output code blocks in your text response as a substitute for editing files
- If the task requires changing a file → call replace_lines or write_file, period
- If the task requires creating a file → call create_file or write_file
- Your final text response should describe what you did (e.g. "Updated foo.py: added X"), \
not reproduce the code you just wrote
- Producing a code snippet in your response without calling a file tool means the task \
is incomplete — the file on disk has not changed

CRITICAL — verify before marking done:
- After editing Python files: run `python -m py_compile <file>` to confirm no syntax errors
- After editing multiple files: run `python -c "import <module>"` to catch import errors
- If the project has tests: run them with run_command after your changes
- Only report success after the verification step passes — never assume edits are correct
""",

    "tester": """\
You are a Senior QA Engineer / Software Tester embedded in this workspace.

Responsibilities:
- Analyse code for correctness, edge cases, and failure modes
- Write or improve tests: unit, integration, and edge cases
- Run existing tests and interpret results
- Report concrete issues with file paths and line numbers

Rules:
- Read the implementation before writing tests
- Prefer tests that catch real bugs over trivial happy-path checks
- Run the test suite after writing new tests to confirm they pass

CRITICAL — use tools to write tests; do not print them:
- NEVER output test code in your text response instead of writing it to a file
- Write tests with write_file or edit_file, then run them with run_command
- Your final response should summarise what tests were added and their results
""",

    "reviewer": """\
You are a Senior Code Reviewer embedded in this workspace.

Your role is to perform rigorous code reviews on changes, Pull Requests, or specific files \
to identify bugs, security vulnerabilities, and maintainability issues.

Responsibilities:
- Review changes for logic errors, race conditions, and edge-case bugs
- Identify security vulnerabilities (injection, improper auth, insecure defaults, memory leaks)
- Evaluate code quality against best practices and project conventions
- Provide clear, actionable findings with specific file paths and line numbers

Constraints:
- YOU CANNOT EDIT FILES — you are strictly read-only
- Your output goes to the Engineer who will implement fixes; do not suggest "fix it yourself"
- Describe exactly what needs to change so the Engineer can implement it

CRITICAL — targeted reads only:
- The system prompt already contains outlines for every source file — do NOT call read_file_outline on indexed files
- Use read_symbol(path, symbol) to read specific functions/classes you want to inspect
- Use read_file with start_line/end_line for focused line ranges
- Use grep_content to find all usages of a pattern across files
- NEVER call read_file on a whole file — use read_symbol or ranged reads only

Rules:
- Be critical but constructive
- Explain the "why" and "how" of each bug or vulnerability you find
- If the code looks good, explicitly state it passes review
- Group findings by severity: Critical → High → Medium → Low
""",

    "general": """\
You are a helpful assistant for the Local Code Agent (lca) CLI tool and general \
programming questions.

You can answer:
- How lca works: multi-agent routing, specialists, caching, scratchpad, settings
- General coding questions, explanations, algorithms, language features
- Conversational questions ("what did you just do?", "explain that", "hey")
- Quick lookups that don't require touching files

Rules:
- Answer directly from knowledge — do NOT call read_file or grep unless the user
  explicitly asks about specific file contents
- Use web_search only for current events or external docs not covered above
- Keep answers concise; code examples only when they add clarity
""",
}

_ARCHITECT_ALLOWED = {
    "read_file_outline", "read_file", "read_symbol", "list_directory",
    "search_files", "grep_content", "web_search", "web_fetch",
    "scratch_write", "scratch_read",
}
_REVIEWER_ALLOWED = {
    "read_file_outline", "read_file", "read_symbol", "list_directory",
    "search_files", "grep_content", "scratch_write", "scratch_read",
}
_GENERAL_ALLOWED = {
    "web_search", "web_fetch", "scratch_write", "scratch_read",
}
_SPECIALIST_TOOLS: dict[str, list] = {
    "architect": [t for t in TOOL_DEFINITIONS if t["function"]["name"] in _ARCHITECT_ALLOWED],
    "engineer":  TOOL_DEFINITIONS,
    "tester":    TOOL_DEFINITIONS,
    "reviewer":  [t for t in TOOL_DEFINITIONS if t["function"]["name"] in _REVIEWER_ALLOWED],
    "general":   [t for t in TOOL_DEFINITIONS if t["function"]["name"] in _GENERAL_ALLOWED],
}
# ── Router ────────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
You are a routing agent. Classify the user's request, pick the right specialist(s), \
and decide which model tier each one should use. Output ONLY a JSON object.

Specialists:
- "engineer"   — DEFAULT choice. Writing/editing code, fixing bugs, implementing \
  features, refactoring. Use this for almost everything involving code changes.
- "tester"     — Writing tests, checking coverage, finding edge cases. Use ONLY when \
  the user explicitly asks for tests or QA.
- "architect"  — Use ONLY when the user explicitly asks to design a new system, \
  plan a major component from scratch, or decide high-level architecture. \
  Do NOT use for normal feature requests, even complex ones.
- "reviewer"   — Code review: checking a PR, diff, or file for bugs and security issues. \
  Read-only. Use when user asks to review, audit, or check code quality.
- "general"    — Conversational questions, lca CLI how-to, explanations, greetings, \
  anything that does NOT require touching files.

Model tiers:
- "main"  — complex reasoning: multi-file refactors, deep debugging, ambiguous requirements
- "fast"  — simple tasks: single-function edits, docstrings, renaming, obvious fixes, \
  all "general" queries
"""

_ROUTER_PROMPT = """\
Output a JSON object with exactly these fields:
{
  "specialist": "architect" | "engineer" | "tester" | "reviewer" | "general",
  "chain": null | ["architect", "engineer"] | ["engineer", "tester"] | ["reviewer", "engineer"],
  "reason": "<one sentence>",
  "refined_task": "<user task rephrased clearly for the specialist>",
  "models": {"<role>": "main" | "fast"}
}

models must have one key per role in the chain (or just the specialist if chain is null).

Chain rules (be conservative — most tasks need only ONE specialist):
- Use ["architect", "engineer"] ONLY if the user explicitly asks to design AND implement \
  something brand new
- Use ["engineer", "tester"] ONLY if the user explicitly asks to implement AND write tests
- Default to chain: null and route to "engineer" for all other coding tasks
- NEVER include "architect" unless the word "design", "architect", or "plan from scratch" \
  appears in the request or there is clearly no existing code to modify

When "chain" is set it overrides "specialist".
Use "general" for greetings, lca how-to questions, and anything that doesn't need file access."""

_ROUTER_FORMAT = {
    "type": "object",
    "properties": {
        "specialist": {"type": "string", "enum": ["architect", "engineer", "tester", "reviewer", "general"]},
        "chain": {
            "oneOf": [
                {"type": "null"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "reason": {"type": "string"},
        "refined_task": {"type": "string"},
        "models": {"type": "object"},
    },
    "required": ["specialist", "chain", "reason", "refined_task", "models"],
}

_VALID_ROLES = set(_ROLE_PROMPTS.keys())

# Patterns that indicate the previous specialist expressed uncertainty or needs
# more information before the next specialist should proceed blindly.
_UNCERTAINTY_RE = re.compile(
    r"\b(unclear|not sure|need(s)? to know|need(s)? more|depends on|"
    r"requires clarification|cannot determine|insufficient( info)?|"
    r"more information (is )?needed|open question|unknown)\b",
    re.I,
)


# ── Fast-path router (no LLM call) ────────────────────────────────────────────
# Patterns are checked in order; first match wins.
# Only used when the intent is unambiguous — unclear messages fall through to LLM.

_FAST_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    # general: greetings and conversational openers → fast
    (re.compile(
        r"^(hey|hi|hello|sup|yo|howdy|hiya)[!?.,\s]*$"
        r"|^(what('s| is) (up|good)|how are you|good (morning|afternoon|evening))[!?.,\s]*$",
        re.I,
    ), "general", "greeting", "fast"),

    # general: questions about lca itself → fast
    (re.compile(
        r"\b(how (do|does|can|should) (i|we|you)|what (is|are|does)|explain|tell me|"
        r"show me|help me understand)\b.{0,60}"
        r"\b(lca|this (cli|tool|agent)|the (agent|cli|tool)|toggle|setting|command|"
        r"specialist|model|cache|scratchpad|compress|fast mode|history|tokens)\b",
        re.I,
    ), "general", "lca how-to question", "fast"),

    # architect: ONLY explicit design/architecture requests (very narrow).
    # Must come BEFORE the general "question" pattern so that
    # "can you design ..." is not swallowed by the general fast-path.
    (re.compile(
        r"\b(architect|blueprint)\b"
        r"|\bdesign (a |the |new |from scratch\b)"
        r"|\bplan (a |the |new |from scratch\b)"
        r"|\bfrom scratch\b",
        re.I,
    ), "architect", "explicit design/architecture request", "main"),

    # tester: explicit test-writing requests → fast (usually templated work)
    (re.compile(
        r"\b(write|add|create|generate)\b.{0,40}\btest(s|ing|cases?)?\b"
        r"|\bunit test|integration test|test coverage\b",
        re.I,
    ), "tester", "explicit test request", "fast"),

    # general: pure questions with no file/code verbs → fast.
    # Negative lookahead excludes design/architect/implement so those fall
    # through to the architect/engineer patterns above.
    (re.compile(
        r"^(what|why|how|when|where|who|which|can you|could you|would you|is it|are there)\b"
        r"(?!.{0,80}\b(design|architect|blueprint|plan|build|implement|"
        r"edit|write|create|delete|rename|refactor|fix|add|update|"
        r"change|modify|read|open|run|test)\b)",
        re.I,
    ), "general", "question without file ops", "fast"),

    # engineer: "add X to Y", "update X in Y", "change X to Y" → fast for targeted edits
    (re.compile(
        r'^(add|update|change|rename|move|delete|remove|fix)\s+\w.{0,60}\s+(in|to|from|of)\s+\w',
        re.I,
    ), "engineer", "targeted file operation", "fast"),

    # general: "what does X do", "why does X", "explain X", "show me X" with no file-op verbs
    (re.compile(
        r'^(what does|why does|why is|what is|explain|show me|tell me)\b'
        r'(?!.{0,60}\b(edit|write|create|delete|fix|implement|add|change|update|run)\b)',
        re.I,
    ), "general", "explanation question", "fast"),

    # general: single-word or very short messages that are clearly conversational
    (re.compile(
        r'^(ok|okay|yes|no|sure|got it|thanks|thank you|cool|great|done|continue|go ahead|proceed)[!?.,\s]*$',
        re.I,
    ), "general", "conversational reply", "fast"),

    # engineer: simple one-liner changes → fast
    (re.compile(
        r"\b(rename|delete|remove|add docstring|add comment|format|"
        r"fix typo|fix import|add type hint)\b",
        re.I,
    ), "engineer", "simple edit", "fast"),

    # reviewer: explicit review, audit, security, or bug-hunting requests → main
    (re.compile(
        r"\b(review|audit|security|vuln|vulnerability|bug hunt|code review)\b",
        re.I,
    ), "reviewer", "explicit review request", "main"),

    # engineer: complex changes → main
    (re.compile(
        r"\b(refactor|implement|debug|fix|migrate|optimi[sz]e|"
        r"rewrite|create|add|update|change|modify|edit)\b",
        re.I,
    ), "engineer", "implementation request", "main"),
]


def _try_fast_route(user_message: str) -> dict | None:
    """Return a routing dict without an LLM call, or None if intent is ambiguous."""
    for pattern, specialist, reason, model_tier in _FAST_PATTERNS:
        if pattern.search(user_message):
            return {
                "specialist": specialist,
                "chain": None,
                "reason": f"fast-path: {reason}",
                "refined_task": user_message,
                "models": {specialist: model_tier},
            }
    return None


def _route(
    user_message: str,
    history: list,
    session: SessionStats,
    on_llm_start: Callable | None = None,
    on_llm_end: Callable | None = None,
    use_fast_path: bool = True,
) -> dict:
    if use_fast_path:
        fast = _try_fast_route(user_message)
        if fast:
            return fast

    messages = (
        [{"role": "system", "content": _ROUTER_SYSTEM}]
        + history
        + [
            {"role": "user", "content": user_message},
            {"role": "user", "content": _ROUTER_PROMPT},
        ]
    )
    if on_llm_start:
        on_llm_start("route")
    raw = ollama_client.chat(messages=messages, tools=[], stream=False, fmt=_ROUTER_FORMAT, model=config.MODEL)
    if on_llm_end:
        on_llm_end()

    session.add(_extract_stats("route", raw))

    content = raw.get("message", {}).get("content", "").strip()
    try:
        route = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        route = {
            "specialist": "engineer",
            "chain": None,
            "reason": "router parse error — defaulting to engineer",
            "refined_task": user_message,
        }
    return route


# ── Specialist runner ─────────────────────────────────────────────────────────

def _specialist_system_prompt(role: str) -> str:
    base = get_base_prompt() if role == "general" else get_system_prompt()
    prompt = base + "\n\n## Your role\n\n" + _ROLE_PROMPTS[role]
    pad = get_context_for_role(role)
    if pad:
        prompt += "\n\n" + pad
    return prompt


def _run_specialist(
    role: str,
    task: str,
    history: list,
    session: SessionStats,
    prior_output: str | None = None,
    on_think: Callable | None = None,
    on_tool_call: Callable | None = None,
    on_tool_result: Callable | None = None,
    on_act_stats: Callable | None = None,
    on_llm_start: Callable | None = None,
    on_llm_end: Callable | None = None,
    on_permission_request: Callable | None = None,
    on_stream_token: Callable | None = None,
    max_iterations: int = 20,
    enable_compression: bool = True,
    enable_fast_route: bool = True,
    model: str | None = None,
) -> str:
    system_prompt = _specialist_system_prompt(role)
    tool_defs = _SPECIALIST_TOOLS[role]

    effective_task = task
    if prior_output:
        uncertainty_notice = ""
        if _UNCERTAINTY_RE.search(prior_output):
            uncertainty_notice = (
                "\nNOTE: The previous specialist's output contains open questions or uncertainties "
                "(marked above). If any are blocking, surface them to the user before proceeding. "
                "Otherwise, state your assumptions explicitly and continue.\n"
            )
        effective_task = (
            f"{task}\n\n"
            f"── Output from previous specialist ──\n"
            f"{prior_output}\n"
            f"── End of previous output ──"
            f"{uncertainty_notice}"
        )

    ck = f"{role}:{model}"
    cached_plan = _should_skip_think(effective_task, ck)
    if cached_plan is not None:
        if on_think:
            from agent import _extract_stats, StepStats
            on_think(cached_plan, StepStats(step="think", prompt_tokens=0, gen_tokens=0, duration_ms=0, tokens_per_sec=0.0))
        plan = cached_plan
    else:
        plan = _think(
            effective_task, history, session,
            on_think=on_think,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
            system_prompt=system_prompt,
            model=model,
            cache_key=ck,
        )

    plan_cap = max(400, config.NUM_CTX // 32)
    plan_for_act = plan[:plan_cap] + ("\n…(truncated)" if len(plan) > plan_cap else "")

    return _act(
        effective_task,
        plan_for_act,
        history,
        session,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_act_stats=on_act_stats,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
        on_permission_request=on_permission_request,
        on_stream_token=on_stream_token,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
        enable_compression=enable_compression,
        model=model,
        specialist=role,
    )


# ── Prefetch helper ───────────────────────────────────────────────────────────

_FILE_MENTION_RE = re.compile(
    r'\b[\w/.-]+\.(?:py|js|ts|go|rs|java|rb|cpp|c|h|md|txt|yaml|yml|json|toml)\b'
)


def _prefetch_mentioned_files(message: str) -> None:
    """Speculatively outline files mentioned in the message (up to 5).
    Errors are silently ignored — this is best-effort prefetching.
    """
    mentioned = _FILE_MENTION_RE.findall(message)
    seen: list[str] = []
    for f in mentioned:
        if f not in seen:
            seen.append(f)
        if len(seen) >= 5:
            break
    for f in seen:
        try:
            execute_tool("read_file_outline", {"path": f})
        except Exception:
            pass


# ── Public API ────────────────────────────────────────────────────────────────

def run_multi_agent(
    user_message: str,
    history: list,
    session: SessionStats | None = None,
    on_route: Callable[[list[str], str], None] | None = None,
    on_specialist_start: Callable[[str], None] | None = None,
    on_think: Callable[[str, StepStats], None] | None = None,
    on_tool_call: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str], None] | None = None,
    on_act_stats: Callable[[StepStats], None] | None = None,
    on_llm_start: Callable[[str], None] | None = None,
    on_llm_end: Callable[[], None] | None = None,
    on_permission_request: Callable[[str, dict, str], bool] | None = None,
    on_stream_token: Callable[[str], None] | None = None,
    max_iterations: int = 20,
    enable_compression: bool = True,
    enable_fast_route: bool = True,
) -> tuple[str, list]:
    """
    Run a full multi-agent cycle: route → specialist(s) → answer.

    on_route(chain, reason)   — called once with the routing decision.
    on_specialist_start(role) — called before each specialist begins.
    enable_compression        — compress stale tool results mid-act (setting: compress).
    enable_fast_route         — skip LLM router for obvious requests (setting: fast).
    All other callbacks mirror run_agent.
    """
    if session is None:
        session = SessionStats()

    history.append({"role": "user", "content": user_message})

    # Speculatively outline files mentioned in the message while routing runs
    threading.Thread(
        target=_prefetch_mentioned_files, args=(user_message,), daemon=True
    ).start()

    # Compress old turns before routing so the router sees a bounded context
    history[:-1] = maybe_compress_history(
        history[:-1],
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
    )

    route = _route(
        user_message, history[:-1], session,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
        use_fast_path=enable_fast_route,
    )

    chain = route.get("chain") or [route.get("specialist", "engineer")]
    chain = [r for r in chain if r in _VALID_ROLES] or ["engineer"]
    refined_task = route.get("refined_task") or user_message
    reason = route.get("reason", "")

    if on_route:
        on_route(chain, reason, {r: config.MODEL for r in chain})

    prior_output: str | None = None
    final_answer = ""

    for role in chain:
        resolved_model = config.MODEL
        if on_specialist_start:
            on_specialist_start(role, resolved_model)

        final_answer = _run_specialist(
            role=role,
            task=refined_task,
            history=history[:-1],
            session=session,
            prior_output=prior_output,
            on_think=on_think,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_act_stats=on_act_stats,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
            on_permission_request=on_permission_request,
            on_stream_token=on_stream_token,
            max_iterations=max_iterations,
            enable_compression=enable_compression,
            model=resolved_model,
        )
        prior_output = final_answer

    history.append({"role": "assistant", "content": final_answer})
    return final_answer, history


def get_specialist_names() -> list[str]:
    return list(_ROLE_PROMPTS.keys())
