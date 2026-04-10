import ast
import os
import re
import subprocess
import time
from pathlib import Path

import config
from config import WORKING_DIR
from tools.cache import (
    file_cache, outline_cache, grep_cache, search_cache,
    mark_workspace_dirty, is_search_stale,
)

def _read_file_max_lines() -> int:
    """Max lines returned by read_file, scaled to context window size.

    At 32k:  500 lines   At 128k: 2048 lines   At 8k: 500 lines (floor)
    """
    return max(500, config.NUM_CTX // 64)


def _resolve(path: str) -> Path:
    """Resolve path and enforce it stays within WORKING_DIR (sandbox)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORKING_DIR) / p
    resolved = p.resolve()
    workspace = Path(WORKING_DIR).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PermissionError(
            f"Path '{path}' resolves outside the workspace ({workspace}). "
            "Access to files outside the working directory is not allowed."
        )
    return resolved


def _safe_resolve(path: str) -> tuple[Path | None, str | None]:
    """Resolve path returning (Path, None) on success or (None, error_str) on sandbox violation."""
    try:
        return _resolve(path), None
    except PermissionError as e:
        return None, f"Error: {e}"


def _mtime_ns(p: Path) -> int | None:
    """Return st_mtime_ns or None if stat fails."""
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


# ── Read ──────────────────────────────────────────────────────────────────────

def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"

    mtime = _mtime_ns(p)
    if mtime is None:
        return f"Error: could not stat file: {path}"

    cache_key = (str(p), mtime, start_line, end_line)
    cached, hit = file_cache.get(cache_key)
    if hit:
        return cached

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total = len(all_lines)

        if start_line is not None or end_line is not None:
            lo = max(1, start_line or 1)
            hi = min(total, end_line or total)
            lines = all_lines[lo - 1:hi]
            numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, lo))
            result = (
                f'<file path="{path}" lines="{total}" showing="{lo}-{hi}">\n'
                f"{numbered}\n</file>"
            )
        elif total > _read_file_max_lines():
            # File too large for full read — return first chunk with explicit signal
            max_lines = _read_file_max_lines()
            visible = all_lines[:max_lines]
            numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(visible))
            result = (
                f'<file path="{path}" lines="{total}" truncated="true" '
                f'showing="1-{max_lines}" '
                f'hint="File has {total} lines. Use start_line/end_line to read other sections.">\n'
                f"{numbered}\n</file>"
            )
        else:
            numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(all_lines))
            result = f'<file path="{path}" lines="{total}">\n{numbered}\n</file>'

        file_cache.put(cache_key, result)
        return result
    except Exception as e:
        return f"Error reading file: {e}"


# ── Write / Edit / Create / Delete ────────────────────────────────────────────

def write_file(path: str, content: str) -> str:
    p, err = _safe_resolve(path)
    if err:
        return err
    # Guard: refuse to massively shrink an existing file — this almost always
    # means the model is replacing a large file with a tiny stub.
    # Threshold: new content must be at least 40% of the existing byte count.
    if p.exists() and p.is_file():
        try:
            existing_size = p.stat().st_size
            new_size = len(content.encode("utf-8"))
            if existing_size > 500 and new_size < existing_size * 0.4:
                return (
                    f"Error: write_file refused — new content ({new_size} bytes) is less than "
                    f"40% of the existing file ({existing_size} bytes). This would destroy most "
                    f"of the file. Use edit_file or replace_lines to make targeted changes instead."
                )
        except OSError:
            pass
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        mark_workspace_dirty()
        lines = content.count("\n") + 1
        return f"Written {lines} lines to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines start_line..end_line (1-indexed, inclusive) with new_content.

    Preferred over edit_file when you know the line numbers from read_file_outline
    — no exact string matching required, so no retry failures.
    """
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)
        if start_line < 1 or end_line < start_line or start_line > total:
            return (
                f"Error: line range {start_line}-{end_line} is out of bounds "
                f"(file has {total} lines)."
            )
        end_line = min(end_line, total)
        replacement = new_content
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"
        new_lines = replacement.splitlines(keepends=True)
        lines[start_line - 1 : end_line] = new_lines
        p.write_text("".join(lines), encoding="utf-8")
        mark_workspace_dirty()
        return (
            f"Replaced lines {start_line}-{end_line} in {path} "
            f"({len(new_lines)} new lines, file now {len(lines)} lines)."
        )
    except Exception as e:
        return f"Error replacing lines: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}. Verify the exact text including whitespace."
        if count > 1:
            return (
                f"Error: old_string appears {count} times in {path}. "
                "Provide more surrounding context to make it unique."
            )
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        mark_workspace_dirty()
        return f"Edited {path}: replaced 1 occurrence."
    except Exception as e:
        return f"Error editing file: {e}"


def create_file(path: str, content: str) -> str:
    p, err = _safe_resolve(path)
    if err:
        return err
    if p.exists():
        return f"Error: file already exists: {path}. Use write_file to overwrite."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        mark_workspace_dirty()
        lines = content.count("\n") + 1
        return f"Created {path} ({lines} lines)"
    except Exception as e:
        return f"Error creating file: {e}"


def delete_file(path: str) -> str:
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    try:
        p.unlink()
        mark_workspace_dirty()
        return f"Deleted {path}"
    except Exception as e:
        return f"Error deleting file: {e}"


# ── Directory / Search ────────────────────────────────────────────────────────

def list_directory(path: str = "", recursive: bool = False) -> str:
    if path:
        p, err = _safe_resolve(path)
        if err:
            return err
    else:
        p = Path(WORKING_DIR)
    if not p.exists():
        return f"Error: directory not found: {path}"
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    try:
        results = []
        if recursive:
            for item in sorted(p.rglob("*")):
                rel = item.relative_to(p)
                tag = "/" if item.is_dir() else ""
                results.append(f"  {rel}{tag}")
        else:
            for item in sorted(p.iterdir()):
                rel = item.relative_to(p)
                tag = "/" if item.is_dir() else ""
                results.append(f"  {rel}{tag}")
        header = f"{p}/" if str(p) != WORKING_DIR else f"{p}/ (working dir)"
        return header + "\n" + "\n".join(results) if results else f"{header}\n  (empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str, directory: str = "") -> str:
    if directory:
        base, err = _safe_resolve(directory)
        if err:
            return err
    else:
        base = Path(WORKING_DIR)
    cache_key = (pattern, str(base))
    cached, hit = search_cache.get(cache_key)
    if hit:
        cached_at, result = cached
        if not is_search_stale(cached_at):
            return result

    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            result = f"No files matched '{pattern}' in {base}"
        else:
            lines = [str(m.relative_to(base)) for m in matches]
            result = f"Found {len(matches)} match(es) for '{pattern}':\n" + "\n".join(f"  {l}" for l in lines)
        search_cache.put(cache_key, (time.monotonic(), result))
        return result
    except Exception as e:
        return f"Error searching files: {e}"


# ── Outline helpers ───────────────────────────────────────────────────────────

def _outline_python(content: str, path: str, total_lines: int) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return _outline_generic(content, path, total_lines) + f"\n(Python parse error: {e})"

    items = []

    def _first_doc(node) -> str:
        d = ast.get_docstring(node) or ""
        return d.strip().split("\n")[0][:80] if d else ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            doc = _first_doc(node)
            suffix = f"  # {doc}" if doc else ""
            items.append(f"  {kind:<10} L{node.lineno:<5} {node.name}{suffix}")
        elif isinstance(node, ast.ClassDef):
            doc = _first_doc(node)
            suffix = f"  # {doc}" if doc else ""
            items.append(f"  {'class':<10} L{node.lineno:<5} {node.name}{suffix}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ck = "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                    cdoc = _first_doc(child)
                    csuffix = f"  # {cdoc}" if cdoc else ""
                    items.append(f"    {ck:<10} L{child.lineno:<5} {child.name}{csuffix}")

    header = f'<outline path="{path}" lines="{total_lines}">'
    return header + "\n" + "\n".join(items) + "\n</outline>"


def _outline_js(content: str, path: str, total_lines: int) -> str:
    items = []
    patterns = [
        (r"^\s*(export\s+)?(async\s+)?function\s*\*?\s*(\w+)", "function"),
        (r"^\s*(export\s+)?(abstract\s+)?class\s+(\w+)", "class"),
        (r"^\s*(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(", "arrow fn"),
        (r"^\s*(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?function", "fn expr"),
    ]
    for i, line in enumerate(content.splitlines(), 1):
        for pattern, kind in patterns:
            m = re.match(pattern, line)
            if m:
                name = m.group(3)
                items.append(f"  {kind:<12} L{i:<5} {name}")
                break

    header = f'<outline path="{path}" lines="{total_lines}">'
    return header + "\n" + "\n".join(items) + "\n</outline>"


def _outline_regex(content: str, path: str, total_lines: int, patterns: list[tuple[str, str, int]]) -> str:
    """Generic regex-based outline. patterns: [(regex, kind_label, name_group), ...]"""
    items = []
    for i, line in enumerate(content.splitlines(), 1):
        for pattern, kind, group in patterns:
            m = re.match(pattern, line)
            if m:
                try:
                    name = m.group(group)
                except IndexError:
                    name = m.group(0).strip()[:60]
                items.append(f"  {kind:<12} L{i:<5} {name}")
                break
    header = f'<outline path="{path}" lines="{total_lines}">'
    body = "\n".join(items) if items else "  (no symbols found)"
    return f"{header}\n{body}\n</outline>"


def _outline_go(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*func\s+\(\w+\s+\*?(\w+)\)\s+(\w+)\s*\(", "method",   2),
        (r"^\s*func\s+(\w+)\s*\(",                        "func",     1),
        (r"^\s*type\s+(\w+)\s+struct",                    "struct",   1),
        (r"^\s*type\s+(\w+)\s+interface",                 "interface",1),
        (r"^\s*type\s+(\w+)\s+",                          "type",     1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_rust(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", "fn",        1),
        (r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)",           "struct",    1),
        (r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)",             "enum",      1),
        (r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)",            "trait",     1),
        (r"^\s*(?:pub(?:\([^)]*\))?\s+)?impl(?:<[^>]*>)?\s+(?:[\w:]+\s+for\s+)?(\w+)", "impl", 1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_java(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*(?:(?:public|protected|private|static|abstract|final|synchronized)\s+)*class\s+(\w+)",     "class",     1),
        (r"^\s*(?:(?:public|protected|private|static|abstract|final|synchronized)\s+)*interface\s+(\w+)", "interface", 1),
        (r"^\s*(?:(?:public|protected|private|static|abstract|final|synchronized)\s+)*enum\s+(\w+)",      "enum",      1),
        (r"^\s*(?:(?:public|protected|private|static|abstract|final|synchronized)\s+)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*\{", "method", 1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_kotlin(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*(?:(?:data|abstract|open|sealed|inner|private|protected|public|internal)\s+)*class\s+(\w+)", "class",    1),
        (r"^\s*(?:(?:private|protected|public|internal|override|suspend|inline)\s+)*fun\s+(\w+)",           "fun",      1),
        (r"^\s*(?:(?:private|protected|public|internal)\s+)*object\s+(\w+)",                                "object",   1),
        (r"^\s*(?:(?:private|protected|public|internal)\s+)*interface\s+(\w+)",                             "interface",1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_ruby(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*def\s+(self\.)?(\w+)",    "def",    2),
        (r"^\s*class\s+(\w+)",           "class",  1),
        (r"^\s*module\s+(\w+)",          "module", 1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_shell(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*function\s+(\w+)\s*[\(\{]", "function", 1),
        (r"^(\w+)\s*\(\s*\)\s*[\{\n]",     "function", 1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_c(content: str, path: str, total_lines: int) -> str:
    patterns = [
        (r"^\s*(?:typedef\s+)?(?:struct|union|enum)\s+(\w+)\s*\{", "struct",   1),
        (r"^\s*(?:class)\s+(\w+)",                                   "class",   1),
        # Function definition: return-type name(params) { — avoid matching declarations
        (r"^[\w\s\*:<>]+\b(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{",    "func",    1),
    ]
    return _outline_regex(content, path, total_lines, patterns)


def _outline_json(content: str, path: str, total_lines: int) -> str:
    import json as _json
    try:
        data = _json.loads(content)
    except Exception:
        return _outline_generic(content, path, total_lines)
    header = f'<outline path="{path}" lines="{total_lines}">'
    if isinstance(data, dict):
        items = []
        for k, v in list(data.items())[:60]:
            t = type(v).__name__
            if isinstance(v, (dict, list)):
                t = f"{t}({len(v)})"
            items.append(f"  key          {k!r}: {t}")
        body = "\n".join(items) if items else "  (empty object)"
    elif isinstance(data, list):
        body = f"  array        {len(data)} items"
    else:
        body = f"  {type(data).__name__}  {str(data)[:80]}"
    return f"{header}\n{body}\n</outline>"


def _outline_yaml(content: str, path: str, total_lines: int) -> str:
    """Extract top-level keys and section headers without a YAML parser."""
    items = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        # Top-level key: starts at column 0, ends with ':'
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_\-]*)(\s*:.*)$", stripped)
        if m:
            items.append(f"  key          L{i:<5} {m.group(1)}")
    header = f'<outline path="{path}" lines="{total_lines}">'
    body = "\n".join(items) if items else "  (no top-level keys found)"
    return f"{header}\n{body}\n</outline>"


def _outline_toml(content: str, path: str, total_lines: int) -> str:
    items = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("["):
            items.append(f"  section      L{i:<5} {stripped}")
        elif "=" in stripped and not line[0:1].isspace():
            key = stripped.split("=")[0].strip()
            items.append(f"  key          L{i:<5} {key}")
    header = f'<outline path="{path}" lines="{total_lines}">'
    body = "\n".join(items) if items else "  (no keys found)"
    return f"{header}\n{body}\n</outline>"


def _outline_markdown(content: str, path: str, total_lines: int) -> str:
    """Extract heading hierarchy from a markdown file (lines starting with #)."""
    headers = []
    for i, line in enumerate(content.splitlines(), 1):
        if line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            headers.append(f"  {'#' * depth:<4} L{i:<5} {title}")
    body = "\n".join(headers) if headers else "  (no headings found)"
    note = "\n[documentation file — read source .py files for implementation details]"
    return f'<outline path="{path}" lines="{total_lines}">\n{body}{note}\n</outline>'


def _outline_generic(content: str, path: str, total_lines: int) -> str:
    lines = content.splitlines()
    preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:30]))
    tail = (
        f"\n  ... ({total_lines - 30} more lines — use read_file to see all)"
        if total_lines > 30 else ""
    )
    return f'<outline path="{path}" lines="{total_lines}">\n{preview}{tail}\n</outline>'


def read_file_outline(path: str) -> str:
    """Return a compact structural outline of a file (functions/classes + line numbers).
    Use this on the first pass to understand a file's shape cheaply.
    Call read_file only when you need the full implementation.
    """
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"

    mtime = _mtime_ns(p)
    if mtime is None:
        return f"Error: could not stat file: {path}"

    cache_key = (str(p), mtime)
    cached, hit = outline_cache.get(cache_key)
    if hit:
        return cached

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = content.count("\n") + 1
    suffix = p.suffix.lower()

    if suffix == ".py":
        result = _outline_python(content, path, total_lines)
    elif suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        result = _outline_js(content, path, total_lines)
    elif suffix == ".go":
        result = _outline_go(content, path, total_lines)
    elif suffix == ".rs":
        result = _outline_rust(content, path, total_lines)
    elif suffix == ".java":
        result = _outline_java(content, path, total_lines)
    elif suffix == ".kt":
        result = _outline_kotlin(content, path, total_lines)
    elif suffix == ".rb":
        result = _outline_ruby(content, path, total_lines)
    elif suffix in (".sh", ".bash", ".zsh"):
        result = _outline_shell(content, path, total_lines)
    elif suffix in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"):
        result = _outline_c(content, path, total_lines)
    elif suffix == ".json":
        result = _outline_json(content, path, total_lines)
    elif suffix in (".yaml", ".yml"):
        result = _outline_yaml(content, path, total_lines)
    elif suffix == ".toml":
        result = _outline_toml(content, path, total_lines)
    elif suffix in (".md", ".mdx", ".rst", ".txt"):
        result = _outline_markdown(content, path, total_lines)
    else:
        result = _outline_generic(content, path, total_lines)

    outline_cache.put(cache_key, result)
    return result


# ── Symbol reader ────────────────────────────────────────────────────────────

def _find_symbol_range_python(content: str, symbol: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) 1-based for the named function or class."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    all_lines = content.splitlines()
    total = len(all_lines)

    def end_of_node(node) -> int:
        # ast.end_lineno available in Python 3.8+
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno
        # Fallback: scan forward from start for next top-level dedent
        start = node.lineno
        for i in range(start, total):
            if i >= len(all_lines):
                break
            if i > start and all_lines[i] and all_lines[i][0] not in (" ", "\t", "#", "\n", ""):
                return i  # line before dedent (0-based index = 1-based line - 1)
        return total

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return node.lineno, end_of_node(node)
    return None


def _find_symbol_range_js(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) 1-based by scanning for the symbol name."""
    patterns = [
        re.compile(rf'(?:function\s+{re.escape(symbol)}\s*\(|'
                   rf'(?:const|let|var)\s+{re.escape(symbol)}\s*=\s*(?:async\s+)?(?:function|\())',
                   re.I),
        re.compile(rf'class\s+{re.escape(symbol)}\b', re.I),
    ]
    for i, line in enumerate(lines, 1):
        if any(p.search(line) for p in patterns):
            # Walk forward counting braces to find the end
            depth = 0
            for j in range(i - 1, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if j >= i and depth <= 0:
                    return i, j + 1
            return i, min(i + 60, len(lines))
    return None


def read_symbol(path: str, symbol: str) -> str:
    """Read a specific function or class by name — no need to know line numbers.

    Returns the full source of the named function/class with line numbers.
    Faster than read_file_outline → read_file for targeted lookups.
    """
    p, err = _safe_resolve(path)
    if err:
        return err
    if not p.exists():
        return f"Error: file not found: {path}"

    mtime = _mtime_ns(p)
    cache_key = (str(p), mtime, symbol)
    cached, hit = file_cache.get(cache_key)
    if hit:
        return cached

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = content.splitlines()
    suffix = p.suffix.lower()
    rng: tuple[int, int] | None = None

    if suffix == ".py":
        rng = _find_symbol_range_python(content, symbol)
    elif suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        rng = _find_symbol_range_js(lines, symbol)

    if rng is None:
        # Fallback: grep for the symbol name and return ±20 lines around the first hit
        for i, line in enumerate(lines, 1):
            if symbol in line:
                start = max(1, i - 2)
                end = min(len(lines), i + 20)
                rng = (start, end)
                break

    if rng is None:
        return f"Symbol '{symbol}' not found in {path}."

    start, end = rng
    end = min(end, len(lines))
    numbered = "\n".join(f"{i+1:4d} | {lines[i]}" for i in range(start - 1, end))
    result = (
        f'<symbol path="{path}" name="{symbol}" lines="{start}-{end}">\n'
        f"{numbered}\n</symbol>"
    )
    file_cache.put(cache_key, result)
    return result


# ── Grep ──────────────────────────────────────────────────────────────────────

def _inline_grep_previews(base_output: str, base_path: Path) -> str:
    """Parse grep output, read ±10 lines around first match in top 3 unique files,
    and append as inline previews. Only appends if total preview is under 3000 chars.
    Skips files already in outline_cache (they'll be fast anyway).
    """
    match_re = re.compile(r'^(.+?):(\d+)[:-]')
    seen: dict[str, int] = {}  # path -> first matched line number
    for line in base_output.splitlines():
        m = match_re.match(line)
        if m:
            fpath, lineno = m.group(1), int(m.group(2))
            if fpath not in seen:
                seen[fpath] = lineno
                if len(seen) >= 3:
                    break

    if not seen:
        return base_output

    previews: list[str] = []
    total_preview_chars = 0

    for fpath, lineno in seen.items():
        p = Path(fpath) if Path(fpath).is_absolute() else base_path / fpath
        try:
            mtime = _mtime_ns(p)
            if mtime is None:
                continue
            # Skip if already in outline_cache (cheap anyway)
            ck = (str(p), mtime)
            _, hit = outline_cache.get(ck)
            if hit:
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines()
            total = len(all_lines)
            lo = max(0, lineno - 11)
            hi = min(total, lineno + 10)
            chunk = all_lines[lo:hi]
            numbered = "\n".join(f"{lo + i + 1:4d} | {l}" for i, l in enumerate(chunk))
            preview_block = f"\n--- {fpath} (lines {lo + 1}–{hi}) ---\n{numbered}"
            if total_preview_chars + len(preview_block) > 3000:
                break
            previews.append(preview_block)
            total_preview_chars += len(preview_block)
        except Exception:
            continue

    if not previews:
        return base_output

    return base_output + "\n\n[Inline preview — top matches]" + "".join(previews)


def grep_content(pattern: str, path: str = "", file_glob: str = "", context_lines: int = 2) -> str:
    if path:
        base, err = _safe_resolve(path)
        if err:
            return err
    else:
        base = Path(WORKING_DIR)
    cache_key = (pattern, str(base), file_glob, context_lines)
    cached, hit = grep_cache.get(cache_key)
    if hit:
        cached_at, result = cached
        if not is_search_stale(cached_at):
            return result

    try:
        cmd = ["grep", "-rn", "--color=never", f"-C{context_lines}", pattern]
        if file_glob:
            cmd += ["--include", file_glob]
        cmd.append(str(base))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = proc.stdout.strip()
        if not output:
            result = f"No matches found for '{pattern}'"
        else:
            lines = output.splitlines()
            from config import NUM_CTX
            max_lines = max(200, NUM_CTX // 20)
            if len(lines) > max_lines:
                result = (
                    "\n".join(lines[:max_lines])
                    + f"\n... ({len(lines) - max_lines} more lines truncated)"
                )
            else:
                result = output

        result = _inline_grep_previews(result, base)
        grep_cache.put(cache_key, (time.monotonic(), result))
        return result
    except subprocess.TimeoutExpired:
        return "Error: grep timed out"
    except Exception as e:
        return f"Error: {e}"
