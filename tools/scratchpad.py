import json
import os
from datetime import datetime

# Define the path for persistent storage
PERSIST_FILE = ".lca_scratchpad_state.json"

# --- State Management ---

# This variable will hold the in-memory representation of the scratchpad
_store: dict[str, list[tuple[str, str]]] = {}

def _load_state():
    """Load the scratchpad state from the local JSON file."""
    global _store
    if os.path.exists(PERSIST_FILE):
        try:
            with open(PERSIST_FILE, 'r') as f:
                data = json.load(f)
                # Reconstruct _store, ensuring timestamps remain usable (though simple string conversion is fine for reloading)
                _store = {}
                for k, v in data.items():
                    # Ensure list of tuples structure
                    _store[k] = v
            print(f"[System] Loaded scratchpad state from {PERSIST_FILE}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"[System] Warning: Could not load state from {PERSIST_FILE}: {e}. Starting fresh.")
            _store = {}
    else:
        _store = {}

def _save_state():
    """Save the current scratchpad state to the local JSON file."""
    try:
        # Convert list of tuples (str, str) to list of lists for JSON serialization compatibility
        serializable_store = {k: list(v) for k, v in _store.items()}
        with open(PERSIST_FILE, 'w') as f:
            json.dump(serializable_store, f, indent=2)
        return f"[System] Successfully saved scratchpad state to {PERSIST_FILE}."
    except Exception as e:
        return f"[System] Error saving scratchpad state: {e}"

# Load state immediately upon module import
_load_state()

# section_name → list of (timestamp_str, content) entries
# The global _store dictionary is used by the functions below.


# ── Core operations ───────

def scratch_write(content: str, section: str = "common") -> str:
    """Append a note to a scratchpad section."""
    if not content.strip():
        return "Error: content is empty."
    section = section.strip() or "common"
    _store.setdefault(section, []).append((_ts(), content.strip()))
    count = len(_store[section])
    return f"Noted in [{section}] ({count} entr{'y' if count == 1 else 'ies'} total)."


def scratch_read(section: str = "") -> str:
    """Read one scratchpad section or all sections if section is omitted."""
    section = section.strip()
    if section:
        entries = _store.get(section, [])
        if not entries:
            return f"Scratchpad [{section}] is empty."
        lines = [f"  [{ts}] {note}" for ts, note in entries]
        return f"## Scratchpad: {section}\n" + "\n".join(lines)

    if not _store:
        return "Scratchpad is empty."
    parts = []
    for sec, entries in _store.items():
        if entries:
            lines = [f"  [{ts}] {note}" for ts, note in entries]
            parts.append(f"## Scratchpad: {sec}\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else "Scratchpad is empty."


def scratch_clear(section: str = "") -> str:
    """Clear one section or the entire scratchpad."""
    section = section.strip()
    if section:
        if section not in _store or not _store[section]:
            return f"Scratchpad [{section}] was already empty."
        count = len(_store[section])
        _store[section] = []
        return f"Cleared {count} entr{'y' if count == 1 else 'ies'} from [{section}]."
    total = sum(len(v) for v in _store.values())
    _store.clear()
    return f"Scratchpad cleared ({total} total entries removed)."


# ── Injection helper (used by multi_agent._specialist_system_prompt) ──────────

# Max notes per section injected into the system prompt.
# Older notes are omitted to prevent stale context from contradicting current state.
_MAX_INJECT_NOTES = 10


def get_context_for_role(role: str) -> str:
    """Return formatted scratchpad content relevant to this role.

    Includes the shared "common" section and the role's own section.
    Only the most recent _MAX_INJECT_NOTES entries per section are injected
    to avoid accumulating stale context that contradicts current state.
    Returns an empty string if both are empty (no injection needed).
    """
    parts = []

    common_all = _store.get("common", [])
    common = common_all[-_MAX_INJECT_NOTES:]
    if common:
        header = "### Shared notes (common)"
        if len(common_all) > _MAX_INJECT_NOTES:
            header += f" — last {_MAX_INJECT_NOTES} of {len(common_all)}"
        lines = [f"  [{ts}] {note}" for ts, note in common]
        parts.append(header + "\n" + "\n".join(lines))

    own_all = _store.get(role, [])
    own = own_all[-_MAX_INJECT_NOTES:]
    if own:
        header = f"### Your notes ({role})"
        if len(own_all) > _MAX_INJECT_NOTES:
            header += f" — last {_MAX_INJECT_NOTES} of {len(own_all)}"
        lines = [f"  [{ts}] {note}" for ts, note in own]
        parts.append(header + "\n" + "\n".join(lines))

    if not parts:
        return ""
    return "## Scratchpad\n\n" + "\n\n".join(parts)


def clear_all() -> None:
    """Wipe everything — call on session clear."""
    global _store
    _store.clear()

# --- Lifecycle Hooks for Persistence ---

# This function should be called at program exit to save state.
# We recommend using a context manager or try...finally block in main application logic.
def save_session_state() -> str:
    """Saves the current state of the scratchpad to disk."""
    return _save_state()

# To make this usable, main execution logic must call save_session_state() before exit.