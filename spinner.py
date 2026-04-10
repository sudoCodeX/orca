"""
Animated spinner for LLM wait states.
Runs in a background thread; call stop() or use as a context manager.
"""

import sys
import threading
import itertools
import time

# Braille spinner frames — same as Claude Code uses
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Word sequences that cycle alongside the spinner for each step type
_WORDS = {
    "think": itertools.cycle([
        "gen thinking",
        "gen reasoning",
        "gen planning",
        "reading ctx",
        "gen thinking",
        "mapping tokens",
    ]),
    "act": itertools.cycle([
        "gen working",
        "generating",
        "gen writing",
        "gen tokens",
        "generating",
        "gen acting",
    ]),
}

_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_RESET = "\033[0m"


class Spinner:
    """
    Thread-based spinner. Start with start(), stop with stop().
    Use as a context manager for automatic cleanup.

    step:  "think" | "act" — controls the word cycle shown.
    model: optional model name shown alongside the spinner.
    """

    def __init__(self, step: str = "act", model: str = ""):
        self._step    = step if step in _WORDS else "act"
        self._model   = model
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._words   = _WORDS[self._step]

    def _run(self):
        frames = itertools.cycle(_FRAMES)
        word   = next(self._words)
        tick   = 0
        model_tag = f" {_DIM}[{self._model}]{_RESET}" if self._model else ""
        while not self._stop.is_set():
            frame = next(frames)
            sys.stderr.write(f"\r{_CYAN}{frame}{_RESET} {_DIM}{word}...{_RESET}{model_tag}")
            sys.stderr.flush()
            time.sleep(0.08)
            tick += 1
            if tick % 15 == 0:          # rotate word every ~1.2 s
                word = next(self._words)
        self._clear()

    def _clear(self):
        sys.stderr.write("\r\033[K")    # carriage return + erase line
        sys.stderr.flush()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
