from .file_ops import (
    read_file,
    read_file_outline,
    read_symbol,
    write_file,
    edit_file,
    replace_lines,
    create_file,
    delete_file,
    list_directory,
    search_files,
    grep_content,
)
from .shell import run_command
from .web import web_search, web_fetch
from .scratchpad import scratch_write, scratch_read, scratch_clear

_TOOLS = {
    "read_file": lambda args: read_file(**args),
    "read_file_outline": lambda args: read_file_outline(**args),
    "read_symbol": lambda args: read_symbol(**args),
    "write_file": lambda args: write_file(**args),
    "edit_file": lambda args: edit_file(**args),
    "replace_lines": lambda args: replace_lines(**args),
    "create_file": lambda args: create_file(**args),
    "delete_file": lambda args: delete_file(**args),
    "list_directory": lambda args: list_directory(**args),
    "search_files": lambda args: search_files(**args),
    "grep_content": lambda args: grep_content(**args),
    "run_command": lambda args: run_command(**args),
    "web_search": lambda args: web_search(**args),
    "web_fetch": lambda args: web_fetch(**args),
    "scratch_write": lambda args: scratch_write(**args),
    "scratch_read": lambda args: scratch_read(**args),
    "scratch_clear": lambda args: scratch_clear(**args),
}


# Models sometimes use 'filepath' or 'file_path' instead of the canonical 'path'.
# Normalise before dispatch so a wrong keyword doesn't cause a hard failure.
_PATH_ALIASES = {"filepath", "file_path", "filename"}


def execute_tool(name: str, arguments: dict) -> str:
    handler = _TOOLS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    # Normalise path-like keyword aliases → 'path'
    if "path" not in arguments:
        for alias in _PATH_ALIASES:
            if alias in arguments:
                arguments = {**arguments, "path": arguments.pop(alias)}
                break
    try:
        return handler(arguments)
    except TypeError as e:
        return f"Error: bad arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error executing '{name}': {e}"
