# Agent Instructions

## File editing rules

- **Never use `write_file` on an existing file** unless you are creating the entire file from scratch (e.g. a new file that does not exist yet).
- For any change to an existing file, always use `edit_file` (string replacement) or `replace_lines` (line-range replacement).
- Before editing, read the relevant section with `read_file_outline` or `read_file` first.
- Never overwrite a large file to make a small change — that destroys all other content.

## Tool preference order for modifications

1. `replace_lines` — when you know the line numbers (fastest, no string matching)
2. `edit_file` — when you know the exact string to replace
3. `write_file` — only for brand-new files that do not exist yet
