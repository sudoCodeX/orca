import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Primary model for reasoning/coding.  Fallback: qwen2.5-coder:7b
MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

WORKING_DIR = os.getenv("AGENT_WORKING_DIR", os.getcwd())

# Safety: require confirmation before running shell commands
CONFIRM_SHELL = os.getenv("CONFIRM_SHELL", "true").lower() != "false"

def get_available_models() -> list[str]:
    """Fetch all models currently available on the Ollama server."""
    try:
        from ollama_client import list_models
        return list_models()
    except Exception:
        return []

# Context window size sent to Ollama (num_ctx).
# Gemma4 supports up to 128k. Default Ollama is 2048–4096 which is too small
# for multi-file workspace queries. 32k is a good balance of quality vs speed.
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

_BASE_SYSTEM_PROMPT = """\
You are a personal workspace agent — an expert assistant that can read, search, create, \
edit, and delete files in any workspace: code repositories, personal knowledge bases, \
Obsidian vaults, documentation, notes, or any directory.

## Tools available
- read_file_outline — fast structural scan: functions/classes + line numbers (use FIRST). \
  Supports: Python, JS/TS, Go, Rust, Java, Kotlin, Ruby, Shell, C/C++, JSON, YAML, TOML, Markdown
- read_symbol       — read a specific function/class by name (use when you know the symbol name)
- read_file         — full file contents (use only when you need the implementation)
- write_file        — overwrite or create a file with new content
- replace_lines     — replace a line range (use when you know line numbers — faster than edit_file)
- edit_file         — precise string replacement inside a file (fallback when line numbers unknown)
- create_file       — create a new file
- delete_file       — delete a file
- list_directory    — list files and subdirectories
- search_files      — find files by glob pattern (e.g. **/*.md, **/interview*)
- grep_content      — search inside files by regex pattern
- run_command       — execute a shell command

## Core rules
1. **Search before asking.** Use search_files or grep_content to locate relevant \
files FIRST. Only ask for clarification if nothing relevant exists.
2. **Outline before reading.** When exploring unfamiliar files, call read_file_outline \
first to understand the structure cheaply. Then call read_file only on the specific \
file (or use grep_content for the specific function) you need to modify or cite.
3. **Read before editing.** Always read or outline a file before making changes.
4. **Be specific, not vague.** Quote from files, reference line numbers, cite filenames.
5. **Infer intent from the workspace.** If the workspace is an Obsidian vault, treat \
notes as the primary source of truth. If it is a code repo, treat source files as \
the source of truth.
6. **Prefer source over docs.** In a code repo, read `.py`/`.ts`/`.js` source files \
directly — do NOT read generated docs, architecture markdown, or README files to \
understand the code. Those files may be stale. Go to the source.
"""


def _workspace_snapshot() -> str:
    """
    Build a 3-level directory tree of the working directory.
    - Level 1 & 2: show files and directories
    - Level 3: show directories only (keeps the snapshot from exploding)
    Skips hidden entries (starting with '.').
    Max 20 children per directory to keep token count reasonable.
    """
    root = Path(WORKING_DIR)
    lines = [f"## Workspace: {root}\n"]

    def _walk(path: Path, indent: int):
        if indent > 4:
            return
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return
        visible = [e for e in entries if not e.name.startswith(".")]
        cap = 10
        for item in visible[:cap]:
            pad = "  " * indent
            if item.is_dir():
                lines.append(f"{pad}{item.name}/")
                _walk(item, indent + 1)
            elif indent <= 2:                 # show files at top 2 levels only
                lines.append(f"{pad}{item.name}")
        if len(visible) > cap:
            lines.append(f"{'  ' * indent}… ({len(visible) - cap} more)")

    _walk(root, indent=1)
    return "\n".join(lines)


_system_prompt_cache: str | None = None


def _code_index() -> str:
    """
    Build a structural outline of every source file in the working directory
    (top-level only, non-hidden, known code/config extensions).

    This is injected into the system prompt so the agent already knows every
    function/class/key in the workspace before its first act iteration —
    eliminating the 2-3 read_file_outline exploration calls that used to start
    every task.

    Uses a lazy import of tools.file_ops to avoid circular imports.
    Capped at 60 files to keep the system prompt size bounded.
    """
    root = Path(WORKING_DIR)
    CODE_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
        ".go", ".rs", ".java", ".kt", ".rb",
        ".c", ".cpp", ".h", ".hpp",
        ".json", ".yaml", ".yml", ".toml",
    }
    try:
        from tools.file_ops import read_file_outline  # lazy — avoids circular import
    except Exception:
        return ""

    files = sorted(
        f for f in root.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in CODE_EXTS
    )[:60]

    if not files:
        return ""

    parts = ["## Code index (pre-built outlines — no need to call read_file_outline on these)\n"]
    for f in files:
        try:
            outline = read_file_outline(str(f.relative_to(root)))
            parts.append(outline)
        except Exception:
            pass
    return "\n".join(parts)


def get_system_prompt() -> str:
    """
    Build the full system prompt (cached for the session).
    Workspace snapshot, code index, and CLAUDE.md are read once and reused.
    Call invalidate_system_prompt_cache() if the workspace changes mid-session.
    """
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    prompt = _BASE_SYSTEM_PROMPT + "\n\n" + _workspace_snapshot()

    index = _code_index()
    if index:
        prompt += "\n\n" + index

    claude_md = Path(WORKING_DIR) / "CLAUDE.md"
    if claude_md.is_file():
        content = claude_md.read_text(encoding="utf-8").strip()
        if content:
            prompt += f"\n\n## Project / workspace instructions (CLAUDE.md)\n\n{content}\n"

    _system_prompt_cache = prompt
    return _system_prompt_cache


def get_base_prompt() -> str:
    """System prompt WITHOUT workspace snapshot — for routing, compression, and general queries."""
    return _BASE_SYSTEM_PROMPT


def invalidate_system_prompt_cache() -> None:
    """Force the next get_system_prompt() call to rebuild the workspace snapshot."""
    global _system_prompt_cache
    _system_prompt_cache = None


# Keep SYSTEM_PROMPT as a static alias (used in tests / direct imports)
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT
