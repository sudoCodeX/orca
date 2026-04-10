#!/usr/bin/env python3
# Ensure project dir is on sys.path so sibling modules are always importable
# regardless of which Python version installed the lca entry point.
import sys as _sys
import traceback
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))


import sys
import argparse
import os
import json
import difflib
import shutil
import config
from dataclasses import dataclass, field
from pathlib import Path

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"


def _show_error(exc: Exception) -> None:
    """Print a clean, actionable error message. Only show a traceback for unexpected errors."""
    msg = str(exc)
    # DNS / TCP connection failures
    if "nodename nor servname" in msg or "Name or service not known" in msg:
        print(f"\n{RED}✗ Cannot reach Ollama — hostname not found.{RESET}")
        print(f"  Check that the server is up and DNS is reachable.")
        print(f"  URL: {msg[:120]}\n")
        return
    if "Connection refused" in msg or "ConnectError" in type(exc).__name__:
        print(f"\n{RED}✗ Connection refused by Ollama server.{RESET}")
        print(f"  Make sure Ollama is running and try again.\n")
        return
    if "timed out" in msg.lower() or "TimeoutError" in type(exc).__name__:
        print(f"\n{RED}✗ Ollama request timed out.{RESET}")
        print(f"  The model may be loading or the server is overloaded. Try again.\n")
        return
    if isinstance(exc, RuntimeError) and "Ollama server error" in msg:
        # Friendly error already formatted by ollama_client._raise_friendly
        print(f"\n{RED}✗ {msg}{RESET}\n")
        return
    # Unexpected error — show full traceback
    print(f"\n{RED}Error: {exc}{RESET}\n")
    traceback.print_exc()


def _colorize_diff(diff_lines: list) -> str:
    result = []
    for line in diff_lines:
        stripped = line.rstrip("\n")
        if stripped.startswith(("+++", "---")):
            result.append(f"{DIM}{stripped}{RESET}")
        elif stripped.startswith("@@"):
            result.append(f"{CYAN}{stripped}{RESET}")
        elif stripped.startswith("+"):
            result.append(f"{GREEN}{stripped}{RESET}")
        elif stripped.startswith("-"):
            result.append(f"{RED}{stripped}{RESET}")
        else:
            result.append(f"{DIM}{stripped}{RESET}")
    return "\n".join(result)


def _build_diff_preview(name: str, arguments: dict, working_dir: str) -> str | None:
    """Return a colored diff string for file-mutating tool calls, or None."""
    from pathlib import Path as _P

    def resolve(p: str) -> _P:
        path = _P(p)
        return path if path.is_absolute() else _P(working_dir) / path

    if name == "create_file":
        content = arguments.get("content", "")
        lines = [f"{GREEN}+ {line}{RESET}" for line in content.splitlines()]
        return "\n".join(lines) if lines else None

    if name == "delete_file":
        try:
            content = resolve(arguments.get("path", "")).read_text(encoding="utf-8", errors="replace")
            lines = [f"{RED}- {line}{RESET}" for line in content.splitlines()]
            return "\n".join(lines) if lines else None
        except Exception:
            return None

    if name == "write_file":
        path = resolve(arguments.get("path", ""))
        new_content = arguments.get("content", "")
        old_content = ""
        try:
            if path.exists():
                old_content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile="current", tofile="proposed", n=3,
        ))
        return _colorize_diff(diff) if diff else f"{DIM}(no changes){RESET}"

    if name == "replace_lines":
        path = resolve(arguments.get("path", ""))
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", start)
        new_content = arguments.get("new_content", "")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            old_chunk = "".join(lines[start - 1 : end])
            replacement = new_content if new_content.endswith("\n") else new_content + "\n"
            diff = list(difflib.unified_diff(
                old_chunk.splitlines(keepends=True),
                replacement.splitlines(keepends=True),
                fromfile=f"{path} (lines {start}-{end})", tofile=str(path), n=3,
            ))
            return _colorize_diff(diff) if diff else f"{DIM}(no changes){RESET}"
        except Exception:
            return None

    if name == "edit_file":
        path = resolve(arguments.get("path", ""))
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            new_content = content.replace(old_string, new_string, 1)
            diff = list(difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=str(path), tofile=str(path), n=3,
            ))
            return _colorize_diff(diff) if diff else f"{DIM}(no changes){RESET}"
        except Exception:
            # fallback: just diff the two strings
            diff = list(difflib.unified_diff(
                old_string.splitlines(keepends=True),
                new_string.splitlines(keepends=True),
                fromfile="remove", tofile="insert", n=2,
            ))
            return _colorize_diff(diff) if diff else None

    return None


REVERSE = "\033[7m"   # nano-style highlight for key labels
NORMAL  = "\033[27m"

# ── Settings ──────────────────────────────────────────────────────────────────

_SETTINGS_FILE = Path.home() / ".lca_settings.json"

@dataclass
class Settings:
    confirm:  bool = True   # require y/N before file mutations
    verbose:  bool = True   # show tool result previews
    think:    bool = True   # show the plan/think block
    stats:    bool = True   # show token stats after each call
    compress: bool = True   # compress stale tool results mid-act
    fast:     bool = True   # fast-path routing (skip LLM for obvious requests)
    max_iter: int  = 15     # max tool-call iterations per specialist

    # ordered for display
    _TOGGLES: tuple = field(default=("confirm","verbose","think","stats","compress","fast"), init=False, repr=False)

    def toggle(self, name: str) -> str | None:
        name = name.strip().lower()
        if not hasattr(self, name) or name.startswith("_"):
            return None
        setattr(self, name, not getattr(self, name))
        return name

    def status_pairs(self) -> list[tuple[str, bool]]:
        return [(k, getattr(self, k)) for k in self._TOGGLES]

    def save(self) -> None:
        data: dict = {k: getattr(self, k) for k in self._TOGGLES}
        # also persist model/ctx/max_iter so they survive restarts
        data["model"] = config.MODEL
        data["num_ctx"] = config.NUM_CTX
        data["max_iter"] = self.max_iter
        try:
            _SETTINGS_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            data = json.loads(_SETTINGS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return s
        for k in s._TOGGLES:
            if k in data and isinstance(data[k], bool):
                setattr(s, k, data[k])
        if "model" in data and isinstance(data["model"], str):
            config.MODEL = data["model"]
            os.environ["OLLAMA_MODEL"] = data["model"]
        if "num_ctx" in data and isinstance(data["num_ctx"], int):
            config.NUM_CTX = data["num_ctx"]
            os.environ["OLLAMA_NUM_CTX"] = str(data["num_ctx"])
        if "max_iter" in data and isinstance(data["max_iter"], int) and 5 <= data["max_iter"] <= 50:
            s.max_iter = data["max_iter"]
        return s


# ── Nano-style statusbar ───────────────────────────────────────────────────────

def _key(label: str) -> str:
    """Render a nano-style highlighted key label."""
    return f"{REVERSE}{BOLD} {label} {RESET}{NORMAL}"


def _print_statusbar(settings: Settings, model: str, ctx: int) -> None:
    """Print a two-row nano-style statusbar above the input prompt."""
    width = shutil.get_terminal_size((80, 24)).columns

    # Row 1 — commands
    cmds = [
        ("X", "Exit"), ("C", "Clear"), ("H", "History"),
        ("T", "Tokens"), ("S", "Scratch"), ("K", "Cache"),
        ("G", "Settings"), ("?", "Help"),
    ]
    row1_parts = [f"{_key(k)}{DIM}{label}{RESET}" for k, label in cmds]
    row1 = "  ".join(row1_parts)

    # Row 2 — settings toggles + active model
    toggle_parts = []
    for name, val in settings.status_pairs():
        indicator = f"{GREEN}ON{RESET}" if val else f"{RED}OFF{RESET}"
        toggle_parts.append(f"{DIM}{name}:{RESET}{indicator}")
    model_short = model if len(model) <= 20 else model[:18] + "…"
    toggle_parts.append(f"{DIM}model:{RESET}{CYAN}{model_short}{RESET}")
    toggle_parts.append(f"{DIM}ctx:{RESET}{CYAN}{ctx // 1024}k{RESET}")
    row2 = "  ".join(toggle_parts)

    sep = f"{DIM}{'─' * width}{RESET}"
    print(sep)
    print(row1)
    print(row2)
    print(sep)


def _print_help() -> None:
    width = shutil.get_terminal_size((80, 24)).columns
    print(f"{DIM}{'─' * width}{RESET}")
    rows = [
        ("exit / quit / q",   "Quit the session"),
        ("clear",             "Reset history, caches, and scratchpad"),
        ("history",           "Show conversation turns"),
        ("tokens",            "Show session token usage"),
        ("cache",             "Show cache hit-rate stats"),
        ("scratch [section]", "Print scratchpad (all or one section)"),
        ("set model <name>",      "Switch model"),
        ("set context <n>",       "Set context window size (tokens)"),
        ("set iterations <n>",    "Set max tool-call iterations (5–50)"),
        ("settings",          "List all toggle settings"),
        ("toggle <name>",     "Flip a setting on/off  (confirm, verbose, think, stats, compress, fast)"),
    ]
    for cmd, desc in rows:
        print(f"  {BOLD}{cmd:<26}{RESET} {DIM}{desc}{RESET}")
    print(f"{DIM}{'─' * width}{RESET}\n")


def main():
    # ── 1. Parse CLI args ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="lca",
        description="Local Code Agent — Ollama-powered file editor",
    )
    parser.add_argument(
        "--repo",
        metavar="PATH",
        default=None,
        help="Repo root to work in (default: current directory)",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Ollama model (default: gemma4:e4b)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt before running shell commands",
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        default=None,
        help="Ollama server URL",
    )
    parser.add_argument(
        "--ctx",
        metavar="N",
        type=int,
        default=None,
        help="Context window size (num_ctx). Default: 32768. Gemma4 supports up to 128k.",
    )
    args = parser.parse_args()

    # ── 2. Push overrides into env before importing config ───────────────────
    # AGENT_WORKING_DIR defaults to cwd at invocation time (not install time)
    os.environ.setdefault("AGENT_WORKING_DIR", os.getcwd())
    if args.repo:
        os.environ["AGENT_WORKING_DIR"] = os.path.abspath(args.repo)
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.no_confirm:
        os.environ["CONFIRM_SHELL"] = "false"
    if args.url:
        os.environ["OLLAMA_URL"] = args.url
    if args.ctx:
        os.environ["OLLAMA_NUM_CTX"] = str(args.ctx)

    # ── 3. Lazy imports — config reads env vars on import ────────────────────
    import config
    import ollama_client
    from agent import make_history, SessionStats, StepStats
    from multi_agent import run_multi_agent
    from tools.cache import all_stats as cache_stats, clear_all as cache_clear_all
    from tools.scratchpad import scratch_read, save_session_state, clear_all as scratch_clear_all
    from spinner import Spinner
    from pathlib import Path

    # ── 4. Helpers ───────────────────────────────────────────────────────────
    def print_banner():
        print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════╗{RESET}")
        print(f"{BOLD}{CYAN}║      Local Code Agent  [multi]        ║{RESET}")
        print(f"{BOLD}{CYAN}╚══════════════════════════════════════╝{RESET}")
        print(f"  {DIM}Ollama  :{RESET} {config.OLLAMA_URL}")
        print(f"  {DIM}Repo    :{RESET} {config.WORKING_DIR}")
        claude_md = Path(config.WORKING_DIR) / "CLAUDE.md"
        if claude_md.is_file():
            print(f"  {DIM}CLAUDE.md:{RESET} {GREEN}loaded{RESET}")
        print()

    def check_connection():
        if not ollama_client.check_connection():
            print(f"{RED}✗ Cannot reach Ollama at {config.OLLAMA_URL}{RESET}")
            print("  Make sure Ollama is running and the URL is correct.")
            sys.exit(1)
        models = ollama_client.list_models()
        if config.MODEL not in models:
            available = ", ".join(models) if models else "(none)"
            print(f"{YELLOW}⚠  Model '{config.MODEL}' not found on server.{RESET}")
            print(f"   Available: {available}")
            print(f"   Pull it:   ollama pull {config.MODEL}")
            print()
            ans = input("Continue anyway? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                sys.exit(1)
        print(f"{GREEN}✓ Connected{RESET}\n")

    def _fmt_stats(s: StepStats, session: SessionStats) -> str:
        tps     = f"{s.tokens_per_sec:.0f} tok/s" if s.tokens_per_sec else "—"
        ms      = f"{s.duration_ms:,}ms"
        prefill = f"prefill {s.prefill_ms:,}ms" if s.prefill_ms else ""
        parts = [
            f"ctx {s.prompt_tokens:,} tokens",
            f"gen {s.gen_tokens}",
            tps,
        ]
        if prefill:
            parts.append(prefill)
        parts += [ms, f"session ↑{session.total_prompt_tokens:,}p +{session.total_gen_tokens:,}g"]
        return f"{DIM}  [{s.step}] " + " │ ".join(parts) + RESET

    _SPECIALIST_LABELS = {
        "architect": f"{CYAN}Architect{RESET}",
        "engineer":  f"{GREEN}Engineer{RESET}",
        "tester":    f"{YELLOW}Tester{RESET}",
    }

    def on_route(chain: list, reason: str, models: dict):
        labels = " → ".join(_SPECIALIST_LABELS.get(r, r) for r in chain)
        print(f"  {DIM}routing →{RESET} {labels}  {DIM}({reason}){RESET}")
        print()

    def on_think(plan: str, stats: StepStats):
        # Highlight MODE line, dim the rest
        print(f"{DIM}  ┌─ plan ────────────────────────────────────{RESET}")
        for line in plan.splitlines():
            if line.upper().startswith("MODE:"):
                print(f"  {CYAN}│ {line}{RESET}")
            else:
                print(f"{DIM}  │ {line}{RESET}")
        print(f"{DIM}  └───────────────────────────────────────────{RESET}")
        print(_fmt_stats(stats, session))
        print()

    _FILE_TOOLS = {"read_file", "write_file", "edit_file", "create_file", "delete_file"}
    _PATH_ARGS  = {"path", "file_path"}

    def on_tool_call(name: str, arguments: dict):
        if name in _FILE_TOOLS:
            # Resolve and display the full path prominently
            raw_path = arguments.get("path") or arguments.get("file_path", "")
            from pathlib import Path as _P
            p = _P(raw_path)
            full = str(p if p.is_absolute() else _P(config.WORKING_DIR) / p)
            extra = {k: repr(v)[:50] for k, v in arguments.items() if k not in _PATH_ARGS}
            extra_str = ("  " + ", ".join(f"{k}={v}" for k, v in extra.items())) if extra else ""
            print(f"  {YELLOW}▶ {name}{RESET}  {BOLD}{full}{RESET}{DIM}{extra_str}{RESET}")
        elif name in {"web_search", "web_fetch"}:
            val = arguments.get("query") or arguments.get("url", "")
            print(f"  {YELLOW}▶ {name}{RESET}  {DIM}{val}{RESET}")
        else:
            args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in arguments.items())
            print(f"  {YELLOW}▶ {name}({args_str}){RESET}")

    def on_tool_result(name: str, result: str):
        preview = result[:200].replace("\n", " ↵ ")
        suffix = "…" if len(result) > 200 else ""
        print(f"  {DIM}  → {preview}{suffix}{RESET}")

    def on_act_stats(stats: StepStats):
        print(_fmt_stats(stats, session))

    _spinner: Spinner | None = None
    _current_model: list[str] = [config.MODEL]  # mutable cell so on_specialist_start can update it
    _did_stream: list[bool] = [False]  # True if on_stream_token already printed content this turn

    def on_specialist_start(role: str, model: str):
        _current_model[0] = model
        label = _SPECIALIST_LABELS.get(role, role.upper())
        model_tag = f"{DIM}[{RESET}{CYAN}{model}{RESET}{DIM}]{RESET}"
        bar = "─" * 40
        print(f"{DIM}  ┌─ {bar}{RESET}")
        print(f"  {BOLD}│ {label}  {model_tag}{RESET}")
        print(f"{DIM}  └─ {bar}{RESET}")
        print()

    def on_llm_start(step: str):
        nonlocal _spinner
        label = {"think": "think", "compress": "compressing history"}.get(step, "act")
        _spinner = Spinner(step=label, model=_current_model[0]).start()

    def on_llm_end():
        nonlocal _spinner
        if _spinner:
            _spinner.stop()
            _spinner = None

    def on_stream_token(token: str):
        """Receives content tokens during ACT streaming.

        Stops the spinner on the first token (clearing the spinner line on
        stderr), prints a header once, then writes each token to stdout so
        the response appears live as the model generates it.
        """
        nonlocal _spinner
        if _spinner:
            _spinner.stop()
            _spinner = None
            sys.stdout.write(f"\n{BOLD}{GREEN}AGENT RESPONSE:{RESET}\n{GREEN}")
            sys.stdout.flush()
        _did_stream[0] = True
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_permission_request(name: str, arguments: dict, description: str) -> bool:
        # Stop spinner before prompting so input isn't garbled
        on_llm_end()
        danger = name in ("delete_file", "run_command")
        color = RED if danger else YELLOW
        print(f"\n{color}{BOLD}  Permission required:{RESET}")
        for line in description.splitlines():
            prefix = "  !! " if danger else "  >> "
            print(f"{color}{prefix}{line}{RESET}")
        diff = _build_diff_preview(name, arguments, config.WORKING_DIR)
        if diff:
            print(f"{DIM}  ── diff ──────────────────────────────────────{RESET}")
            for line in diff.splitlines():
                print(f"  {line}")
            print(f"{DIM}  ──────────────────────────────────────────────{RESET}")
        try:
            ans = input(f"  Allow? {BOLD}[y/N]{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        approved = ans in ("y", "yes")
        if not approved:
            print(f"{DIM}  Denied.{RESET}")
        return approved

    # ── 5. REPL ──────────────────────────────────────────────────────────────
    settings = Settings.load()
    print_banner()
    check_connection()

    history = make_history()
    session = SessionStats()

    # Wire settings into callbacks
    def _on_think(plan: str, stats: StepStats):
        if settings.think:
            on_think(plan, stats)

    def _on_tool_result(name: str, result: str):
        if settings.verbose:
            on_tool_result(name, result)

    def _on_act_stats(stats: StepStats):
        if settings.stats:
            on_act_stats(stats)

    def _on_permission_request(name: str, arguments: dict, description: str) -> bool:
        if not settings.confirm:
            # Auto-approve, but still show the diff so the user can see what changed.
            diff = _build_diff_preview(name, arguments, config.WORKING_DIR)
            if diff:
                print(f"{DIM}  ── diff ──────────────────────────────────────{RESET}")
                for line in diff.splitlines():
                    print(f"  {line}")
                print(f"{DIM}  ──────────────────────────────────────────────{RESET}")
            return True
        return on_permission_request(name, arguments, description)

    while True:
        _print_statusbar(settings, config.MODEL, config.NUM_CTX)
        try:
            user_input = input(f"{BOLD}{BLUE}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue

        # parse cmd word separately so `set model foo` works correctly
        parts = user_input.split()
        cmd   = parts[0].lower()
        args  = parts[1:]

        if cmd in ("exit", "quit", "q"):
            print(f"{DIM}Bye!{RESET}")
            save_session_state()
            break

        if cmd == "?" or cmd == "help":
            _print_help()
            continue

        if cmd == "clear":
            history = make_history()
            session = SessionStats()
            cache_clear_all()
            scratch_clear_all()
            print(f"{DIM}History, caches, and scratchpad cleared.{RESET}\n")
            continue

        if cmd == "history":
            if not history:
                print(f"  {DIM}(no history){RESET}\n")
            for msg in history:
                role = msg["role"].upper()
                snippet = (msg.get("content") or "")[:120].replace("\n", " ")
                print(f"  {DIM}{role}: {snippet}{RESET}")
            print()
            continue

        if cmd == "tokens":
            print(
                f"  {DIM}Session: {session.total_calls} LLM calls  │ "
                f"prompt {session.total_prompt_tokens:,} tokens  │ "
                f"generated {session.total_gen_tokens:,} tokens{RESET}\n"
            )
            continue

        if cmd == "cache":
            stats = cache_stats()
            print(f"{DIM}  ── cache stats ──────────────────────────────{RESET}")
            for cname, s in stats.items():
                bar_filled = int(s.hit_rate * 20)
                bar = f"{GREEN}{'█' * bar_filled}{DIM}{'░' * (20 - bar_filled)}{RESET}"
                print(f"  {BOLD}{cname:<8}{RESET} {bar}  {DIM}{s}{RESET}")
            print(f"{DIM}  ────────────────────────────────────────────{RESET}\n")
            continue

        if cmd == "scratch":
            section = args[0] if args else ""
            content = scratch_read(section)
            print(f"{DIM}  ── scratchpad {'(' + section + ') ' if section else ''}──────────────────────────{RESET}")
            for line in content.splitlines():
                print(f"  {DIM}{line}{RESET}")
            print(f"{DIM}  ────────────────────────────────────────────{RESET}\n")
            continue

        if cmd == "settings":
            print(f"{DIM}  ── settings ─────────────────────────────────{RESET}")
            for name, val in settings.status_pairs():
                indicator = f"{GREEN}ON{RESET}" if val else f"{RED}OFF{RESET}"
                print(f"  {BOLD}{name:<12}{RESET} {indicator}")
            print(f"  {BOLD}{'model':<12}{RESET} {CYAN}{config.MODEL}{RESET}")
            print(f"  {BOLD}{'context':<12}{RESET} {CYAN}{config.NUM_CTX:,}{RESET}")
            print(f"  {BOLD}{'max_iter':<12}{RESET} {CYAN}{settings.max_iter}{RESET}")
            print(f"{DIM}  ────────────────────────────────────────────{RESET}\n")
            continue

        if cmd == "toggle":
            if not args:
                print(f"{DIM}Usage: toggle <{' | '.join(settings._TOGGLES)}>{RESET}\n")
                continue
            name = settings.toggle(args[0])
            if name is None:
                print(f"{RED}Unknown setting: {args[0]}{RESET}  (options: {', '.join(settings._TOGGLES)})\n")
            else:
                val = getattr(settings, name)
                indicator = f"{GREEN}ON{RESET}" if val else f"{RED}OFF{RESET}"
                print(f"  {BOLD}{name}{RESET} → {indicator}\n")
                settings.save()
            continue

        if cmd == "set":
            if not args:
                print(f"{DIM}Usage: set <model|context|iterations> [value]{RESET}\n")
                continue
            key, value = args[0].lower(), " ".join(args[1:])

            if key == "model":
                available = config.get_available_models()
                if not available:
                    print(f"{RED}Could not fetch model list from Ollama.{RESET}\n")
                    continue
                if not value:
                    # ── interactive picker ────────────────────────────────
                    print(f"{DIM}  ── available models ──{RESET}")
                    for i, m in enumerate(available, 1):
                        marker = f"{GREEN}*{RESET}" if m == config.MODEL else " "
                        print(f"  {marker} {DIM}{i:>2}.{RESET} {m}")
                    print(f"{DIM}  Enter number or model name (blank to cancel):{RESET} ", end="", flush=True)
                    try:
                        raw = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if not raw:
                        continue
                    if raw.isdigit():
                        idx = int(raw) - 1
                        if 0 <= idx < len(available):
                            value = available[idx]
                        else:
                            print(f"{RED}Invalid number.{RESET}\n")
                            continue
                    else:
                        value = raw
                # ── apply ─────────────────────────────────────────────────
                if value not in available:
                    print(f"{RED}Model '{value}' not found.{RESET}  Available: {', '.join(available[:6])}\n")
                else:
                    config.MODEL = value
                    os.environ["OLLAMA_MODEL"] = value
                    config.invalidate_system_prompt_cache()
                    print(f"  model → {CYAN}{value}{RESET}\n")
                    settings.save()

            elif key == "context":
                _CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072]
                if not value:
                    # ── interactive picker ────────────────────────────────
                    print(f"{DIM}  ── context window presets (current: {CYAN}{config.NUM_CTX:,}{RESET}{DIM}) ──{RESET}")
                    for i, n in enumerate(_CTX_PRESETS, 1):
                        marker = f"{GREEN}*{RESET}" if n == config.NUM_CTX else " "
                        label = f"{n // 1024}k"
                        print(f"  {marker} {DIM}{i}.{RESET} {label:<6} {DIM}({n:,} tokens){RESET}")
                    print(f"  {DIM}  or type a custom value (1024–200000){RESET}")
                    print(f"{DIM}  Enter number or token count (blank to cancel):{RESET} ", end="", flush=True)
                    try:
                        raw = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if not raw:
                        continue
                    if raw.isdigit() and 1 <= int(raw) <= len(_CTX_PRESETS):
                        value = str(_CTX_PRESETS[int(raw) - 1])
                    else:
                        value = raw
                # ── apply ─────────────────────────────────────────────────
                try:
                    n = int(value)
                    if not (1024 <= n <= 200_000):
                        print(f"{RED}Context must be between 1,024 and 200,000.{RESET}\n")
                    else:
                        config.NUM_CTX = n
                        os.environ["OLLAMA_NUM_CTX"] = str(n)
                        config.invalidate_system_prompt_cache()
                        print(f"  context → {CYAN}{n:,}{RESET}\n")
                        settings.save()
                except ValueError:
                    print(f"{RED}Not a valid number: {value!r}{RESET}\n")
            elif key == "iterations":
                if not value:
                    print(f"{DIM}  Current max iterations: {CYAN}{settings.max_iter}{RESET}")
                    print(f"{DIM}  Enter a number (5–50):{RESET} ", end="", flush=True)
                    try:
                        value = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if not value:
                        continue
                try:
                    n = int(value)
                    if not (5 <= n <= 50):
                        print(f"{RED}Iterations must be between 5 and 50.{RESET}\n")
                    else:
                        settings.max_iter = n
                        print(f"  max iterations → {CYAN}{n}{RESET}\n")
                        settings.save()
                except ValueError:
                    print(f"{RED}Not a valid number: {value!r}{RESET}\n")
            else:
                print(f"{RED}Unknown: set {key}{RESET}  (use: model | context | iterations)\n")
            continue

        # ── Agent turn ────────────────────────────────────────────────────────
        print()
        _did_stream[0] = False
        try:
            answer, history = run_multi_agent(
                user_input,
                history,
                session=session,
                on_route=on_route,
                on_specialist_start=on_specialist_start,
                on_think=_on_think,
                on_tool_call=on_tool_call,
                on_tool_result=_on_tool_result,
                on_act_stats=_on_act_stats,
                on_llm_start=on_llm_start,
                on_llm_end=on_llm_end,
                on_permission_request=_on_permission_request,
                on_stream_token=on_stream_token,
                enable_compression=settings.compress,
                enable_fast_route=settings.fast,
                max_iterations=settings.max_iter,
            )
            if _did_stream[0]:
                # Content already printed token-by-token; just close the colour and separator
                print(f"{RESET}\n{BOLD}{GREEN}========================================={RESET}")
            else:
                print(f"\n{BOLD}{GREEN}========================================={RESET}\n{BOLD}{GREEN}AGENT FINAL RESPONSE:{RESET}\n{GREEN}{answer}{RESET}\n{BOLD}{GREEN}========================================={RESET}")
        except KeyboardInterrupt:
            on_llm_end()
            print(f"\n{YELLOW}Interrupted.{RESET}\n")
        except Exception as e:
            on_llm_end()
            _show_error(e)


if __name__ == "__main__":
    main()
