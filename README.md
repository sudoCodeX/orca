# Orca — Local Agentic Coding Assistant

> **Orca** (**O**llama **R**easoning **C**ode **A**gent) is a local, offline-first CLI that talks to your codebase through natural language. It runs entirely on your machine using [Ollama](https://ollama.com/) — no cloud, no API keys, no data leaving your network.

## Demo

<video src="assets/Screen Recording 2026-04-10 at 4.49.52 PM.mp4" controls width="100%"></video>

```
╔══════════════════════════════════════╗
║      Orca  [multi-agent mode]         ║
╚══════════════════════════════════════╝

  ___  _ __ ___ __ _
 / _ \| '__/ __/ _` |
| (_) | | | (_| (_| |
 \___/|_|  \___\__,_|


  Ollama  : http://localhost:11434
  Repo    : /your/project

✓ Connected

 X Exit   C Clear   H History   T Tokens   S Scratch   K Cache   G Settings   ? Help
confirm:ON  verbose:ON  think:ON  stats:ON  compress:ON  fast:ON  model:gemma4:e4b  ctx:32k
────────────────────────────────────────────────────────
you> refactor the auth module to use async/await throughout
```

---

## Features

- **THINK → ACT loop** — the agent plans before acting, showing its reasoning so you can course-correct
- **Multi-agent routing** — architect, engineer, and tester specialists are selected automatically per task
- **Full file toolset** — read, search, edit, create, delete files; grep across the workspace; run shell commands
- **Diff previews** — every file mutation shows a coloured diff before it's applied
- **Scratchpad** — persistent in-session notes the agent uses across turns
- **History compression** — long conversations are summarised automatically to stay within the context window
- **Workspace sandboxing** — file access is restricted to the working directory; path traversal is blocked
- **LRU caching** — file reads, outlines, greps, and web fetches are cached to avoid redundant calls
- **Web tools** — optional DuckDuckGo search and URL fetch (Jina Reader with fallback)

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (or on a remote host you control)

---

## Installation

```bash
git clone https://github.com/yourusername/orca-agent
cd orca-agent
uv sync
uv pip install -e .
```

To make `orca` available globally (without activating the venv each time):

```bash
uv tool install --editable .
```

Then pull a model:

```bash
ollama pull gemma4:e4b
```

---

## Quick Start

```bash
# Run in the current directory
orca

# Run against a specific repo
orca --repo /path/to/your/project

# Choose a model
orca --model gemma4:32b-cloud

# Increase context window (good for large files)
orca --ctx 65536
```

### Example prompts

```
you> explain how the routing system works in multi_agent.py
you> add unit tests for the _resolve() function in tools/file_ops.py
you> find all TODO comments and create a GitHub issues draft for each
you> refactor config.py to load settings lazily
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:12b` | Model name |
| `OLLAMA_NUM_CTX` | `32768` | Context window size (tokens) |
| `AGENT_WORKING_DIR` | current directory | Workspace root (file access is sandboxed here) |
| `CONFIRM_SHELL` | `true` | Require confirmation before running shell commands |

Create a `.env` file in the project root to set these persistently (never commit it).

### Runtime settings

Use the `set` and `toggle` commands inside the REPL:

```
set model qwen2.5-coder:7b   # switch model live
set context 65536             # change context window
set iterations 20             # max tool-call loops per turn
toggle confirm                # flip any setting on/off
settings                      # show all current settings
```

Settings are saved to `~/.orca_settings.json` and restored on next launch.

---

## Project Structure

```
orca-agent/
├── main.py              # CLI entry point, REPL, settings
├── agent.py             # Core THINK → ACT loop, history compression
├── multi_agent.py       # Multi-specialist routing (architect / engineer / tester)
├── config.py            # Environment config, workspace snapshot, system prompt
├── ollama_client.py     # Ollama HTTP client (streaming + structured output)
├── spinner.py           # Terminal spinner for LLM wait states
└── tools/
    ├── definitions.py   # Tool schemas exposed to the model
    ├── file_ops.py      # read, write, edit, create, delete, grep, search, outline
    ├── executor.py      # Tool dispatch
    ├── shell.py         # run_command (with confirmation)
    ├── web.py           # web_search (DuckDuckGo), web_fetch (Jina Reader)
    ├── scratchpad.py    # In-session key-value notepad
    └── cache.py         # LRU caches for file/outline/grep/web results
```

---

## Tested Models

The following models have been tested with Orca. Structured output (JSON) and multi-turn tool use are required.

| Model | Notes |
|---|---|
| `gemma4:e4b` | Default model — good balance of speed and quality |
| `gemma4:32b-cloud` | Best performance; requires Ollama cloud routing |
| `qwen2.5-coder:7b` | Fast local alternative; good for coding tasks |

> **Tip:** Start with `gemma4:e4b`. Use `gemma4:32b-cloud` for complex multi-file reasoning.

Models must support Ollama's tool-use API. Check compatibility at [ollama.com/library](https://ollama.com/library).

---

## Security

### Workspace sandboxing

All file operations are restricted to the `AGENT_WORKING_DIR`. Attempts to read or write files outside the workspace are blocked at the tool level.

### Shell command confirmation

`run_command` prompts for confirmation before execution (`CONFIRM_SHELL=true` by default). Disable only if you fully trust the model and your prompts.

### Web fetch

`web_fetch` blocks requests to private/loopback IP addresses to prevent SSRF. Hostname-based internal addresses (e.g. `myserver.local`) are not blocked — set `OLLAMA_URL` to a non-routable address if you run Ollama on a LAN host.

### What Orca cannot do

- Access the internet without you explicitly asking it to search/fetch
- Run shell commands without confirmation (by default)
- Read or write files outside the working directory
- Authenticate to external services

---

## REPL Commands

| Command | Description |
|---|---|
| `help` / `?` | Show all commands |
| `exit` / `q` | Quit |
| `clear` | Reset history, caches, and scratchpad |
| `history` | Show conversation turns |
| `tokens` | Session token usage |
| `cache` | Cache hit-rate stats |
| `scratch [section]` | Print scratchpad |
| `set model <name>` | Switch model (interactive picker if no name given) |
| `set context <n>` | Set context window size |
| `set iterations <n>` | Max tool-call loops per turn (5–50) |
| `settings` | List all current settings |
| `toggle <name>` | Flip: confirm, verbose, think, stats, compress, fast |

---

## Contributing

Issues and PRs are welcome. Please open an issue first for significant changes.

- Code style: standard Python, no formatter enforced yet
- Test with at least `qwen2.5-coder:7b` before submitting
- Keep tool definitions in `tools/definitions.py` in sync with implementations in `tools/file_ops.py`

---

## License

MIT — see [LICENSE](LICENSE).
