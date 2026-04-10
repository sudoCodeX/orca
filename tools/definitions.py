TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file_outline",
            "description": (
                "Return a compact structural outline of a file: functions and classes "
                "with their line numbers and one-line docstrings. "
                "Use this FIRST when exploring an unfamiliar file — it is much cheaper "
                "than read_file. Only call read_file when you need the full implementation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory or absolute).",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_symbol",
            "description": (
                "Read a specific function or class by name without needing line numbers. "
                "Use this instead of read_file_outline → read_file when you already know "
                "the symbol name you want to inspect or modify. Supports Python (AST-based) "
                "and JavaScript/TypeScript (regex + brace depth). Falls back to grep for "
                "other languages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory or absolute).",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Name of the function or class to read (e.g. 'MyClass', 'handle_request').",
                    },
                },
                "required": ["path", "symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file's contents with line numbers. "
                "Use start_line/end_line to read a specific range (e.g. a single function) "
                "instead of the whole file — much cheaper when you know the line numbers "
                "from read_file_outline. Omit both to read the entire file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory or absolute).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive). Omit to start from line 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive). Omit to read to end of file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, overwriting it if it exists. Use for full rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace a line range in a file with new content. "
                "PREFERRED over edit_file when you know the line numbers from "
                "read_file_outline — no exact string matching, no retry failures. "
                "Use read_file_outline to get line numbers, then call this directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (1-based, inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace (1-based, inclusive).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement text (replaces the entire line range).",
                    },
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a precise string replacement in a file. "
                "old_string must match exactly (including whitespace). "
                "Use replace_lines instead when you know the line numbers — it is "
                "faster and never fails due to whitespace mismatches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file. Fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for the new file."},
                    "content": {
                        "type": "string",
                        "description": "Initial content of the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to delete."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Defaults to working directory.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively. Default false.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Find files matching a glob pattern in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to working directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_content",
            "description": "Search for a regex pattern inside files and return matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search. Defaults to working directory.",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Only search files matching this glob, e.g. '*.py'.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match. Default 2.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return its output. Use for git, tests, installs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Directory to run in. Defaults to working directory.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "Use when the user asks about current events, external documentation, "
                "company info, or anything not in the local workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8, max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return its content as plain text. "
                "Use after web_search to read the full content of a specific page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 8000).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratch_write",
            "description": (
                "Append a note to your scratchpad. Use this to record decisions, "
                "findings, TODOs, or any information you want to remember across "
                "tool calls. Notes in 'common' are visible to all specialists. "
                "Notes in your own section (e.g. 'engineer') are private to you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note to record.",
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Section name: 'common' (shared), 'architect', 'engineer', "
                            "'tester', or any custom name. Defaults to 'common'."
                        ),
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratch_read",
            "description": (
                "Read scratchpad notes. Omit section to read everything. "
                "Note: your own section and 'common' are already shown in your "
                "system prompt — only call this to check another specialist's notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Section to read. Omit to read all sections.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratch_clear",
            "description": "Clear a scratchpad section or the entire scratchpad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Section to clear. Omit to clear everything.",
                    },
                },
                "required": [],
            },
        },
    },
]
