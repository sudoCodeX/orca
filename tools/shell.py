import subprocess
from pathlib import Path

from config import WORKING_DIR, CONFIRM_SHELL


def run_command(command: str, working_dir: str = "") -> str:
    cwd = working_dir if working_dir else WORKING_DIR

    if CONFIRM_SHELL:
        print(f"\n  [shell] $ {command}")
        ans = input("  Run this command? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return "Command cancelled by user."

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.strip())
        if result.stderr.strip():
            parts.append(f"[stderr]\n{result.stderr.strip()}")
        output = "\n".join(parts) if parts else "(no output)"
        if result.returncode != 0:
            output = f"[exit {result.returncode}]\n{output}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60s"
    except Exception as e:
        return f"Error running command: {e}"
