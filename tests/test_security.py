"""
Security and correctness tests for orca-agent.

Covers:
- Path traversal sandbox enforcement (_resolve)
- write_file destructive overwrite guard
- web_fetch SSRF protection
- config defaults
"""
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Point AGENT_WORKING_DIR at a fresh temp directory for every test."""
    monkeypatch.setenv("AGENT_WORKING_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    # Re-import config so it picks up the new env vars
    if "config" in sys.modules:
        del sys.modules["config"]
    if "tools.file_ops" in sys.modules:
        del sys.modules["tools.file_ops"]
    yield tmp_path


# ── Path traversal tests ───────────────────────────────────────────────────────

class TestPathSandbox:
    def _ops(self):
        import tools.file_ops as fo
        return fo

    def test_read_within_workspace(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "hello.txt").write_text("hi")
        result = fo.read_file("hello.txt")
        assert "hi" in result

    def test_read_absolute_within_workspace(self, isolated_workspace):
        fo = self._ops()
        p = isolated_workspace / "hello.txt"
        p.write_text("hi")
        result = fo.read_file(str(p))
        assert "hi" in result

    def test_read_traversal_blocked(self, isolated_workspace):
        fo = self._ops()
        result = fo.read_file("../../../etc/passwd")
        assert "Error" in result or "not allowed" in result

    def test_read_absolute_outside_blocked(self, isolated_workspace):
        fo = self._ops()
        result = fo.read_file("/etc/passwd")
        assert "Error" in result or "not allowed" in result

    def test_write_traversal_blocked(self, isolated_workspace):
        fo = self._ops()
        result = fo.write_file("../evil.txt", "pwned")
        assert "Error" in result or "not allowed" in result
        assert not (isolated_workspace.parent / "evil.txt").exists()

    def test_write_within_workspace(self, isolated_workspace):
        fo = self._ops()
        result = fo.write_file("new.txt", "hello")
        assert "Written" in result
        assert (isolated_workspace / "new.txt").read_text() == "hello"

    def test_edit_file_traversal_blocked(self, isolated_workspace):
        fo = self._ops()
        result = fo.edit_file("../../important.py", "old", "new")
        assert "Error" in result or "not allowed" in result

    def test_delete_traversal_blocked(self, isolated_workspace):
        fo = self._ops()
        result = fo.delete_file("../../../tmp/something")
        assert "Error" in result or "not allowed" in result

    def test_nested_path_within_workspace(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "sub").mkdir()
        (isolated_workspace / "sub" / "file.txt").write_text("ok")
        result = fo.read_file("sub/file.txt")
        assert "ok" in result

    def test_symlink_escape_blocked(self, isolated_workspace):
        """A symlink pointing outside the workspace must be blocked after resolve()."""
        fo = self._ops()
        link = isolated_workspace / "escape_link"
        link.symlink_to("/etc")
        result = fo.read_file("escape_link/passwd")
        assert "Error" in result or "not allowed" in result


# ── Destructive write guard ────────────────────────────────────────────────────

class TestWriteGuard:
    def _ops(self):
        import tools.file_ops as fo
        return fo

    def test_write_new_file_always_allowed(self, isolated_workspace):
        fo = self._ops()
        result = fo.write_file("brand_new.py", "x = 1\n")
        assert "Written" in result

    def test_write_small_content_to_large_file_blocked(self, isolated_workspace):
        fo = self._ops()
        big = "x = 1\n" * 200          # ~1200 bytes
        (isolated_workspace / "big.py").write_text(big)
        result = fo.write_file("big.py", "stub\n")   # 5 bytes — < 40%
        assert "refused" in result.lower()
        # Original file must be untouched
        assert (isolated_workspace / "big.py").read_text() == big

    def test_write_adequate_replacement_allowed(self, isolated_workspace):
        fo = self._ops()
        original = "x = 1\n" * 10
        (isolated_workspace / "small.py").write_text(original)
        replacement = "x = 2\n" * 8    # 80% of original — above threshold
        result = fo.write_file("small.py", replacement)
        assert "Written" in result

    def test_write_tiny_file_replacement_always_allowed(self, isolated_workspace):
        """Files under 500 bytes are exempt from the size guard."""
        fo = self._ops()
        (isolated_workspace / "tiny.txt").write_text("abc")
        result = fo.write_file("tiny.txt", "x")
        assert "Written" in result


# ── SSRF protection ───────────────────────────────────────────────────────────

class TestSSRF:
    def _is_private(self, url):
        from tools.web import _is_private_url
        return _is_private_url(url)

    def test_loopback_blocked(self):
        assert self._is_private("http://127.0.0.1/admin")

    def test_loopback_alt_blocked(self):
        assert self._is_private("http://127.0.0.2:8080/secret")

    def test_rfc1918_10_blocked(self):
        assert self._is_private("http://10.0.0.1/internal")

    def test_rfc1918_172_blocked(self):
        assert self._is_private("http://172.16.5.1/api")

    def test_rfc1918_192_blocked(self):
        assert self._is_private("http://192.168.1.1/router")

    def test_link_local_blocked(self):
        assert self._is_private("http://169.254.169.254/latest/meta-data/")  # AWS IMDS

    def test_public_ip_allowed(self):
        assert not self._is_private("http://8.8.8.8/")

    def test_public_hostname_allowed(self):
        assert not self._is_private("https://github.com/repo")

    def test_web_fetch_loopback_returns_error(self, monkeypatch):
        from tools import web
        result = web.web_fetch("http://127.0.0.1:8080/secret")
        assert "not allowed" in result.lower() or "Error" in result


# ── Config defaults ────────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_default_ollama_url_is_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_URL", raising=False)
        if "config" in sys.modules:
            del sys.modules["config"]
        import config
        assert "localhost" in config.OLLAMA_URL
        assert "fsocity" not in config.OLLAMA_URL

    def test_no_fast_model(self):
        if "config" in sys.modules:
            del sys.modules["config"]
        import config
        assert not hasattr(config, "FAST_MODEL"), \
            "FAST_MODEL was removed — it should not exist on config"


# ── File ops basic correctness ─────────────────────────────────────────────────

class TestFileOps:
    def _ops(self):
        import tools.file_ops as fo
        return fo

    def test_read_nonexistent_returns_error(self, isolated_workspace):
        fo = self._ops()
        result = fo.read_file("nope.txt")
        assert "Error" in result

    def test_edit_file_applies_replacement(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "f.py").write_text("foo = 1\nbar = 2\n")
        result = fo.edit_file("f.py", "foo = 1", "foo = 99")
        assert "Error" not in result
        assert (isolated_workspace / "f.py").read_text() == "foo = 99\nbar = 2\n"

    def test_edit_file_missing_old_string(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "f.py").write_text("hello\n")
        result = fo.edit_file("f.py", "NOTHERE", "x")
        assert "Error" in result or "not found" in result.lower()

    def test_replace_lines(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "f.py").write_text("a\nb\nc\n")
        result = fo.replace_lines("f.py", 2, 2, "REPLACED\n")
        assert "Error" not in result
        assert (isolated_workspace / "f.py").read_text() == "a\nREPLACED\nc\n"

    def test_create_and_delete_file(self, isolated_workspace):
        fo = self._ops()
        fo.create_file("temp.txt", "content")
        assert (isolated_workspace / "temp.txt").exists()
        fo.delete_file("temp.txt")
        assert not (isolated_workspace / "temp.txt").exists()

    def test_grep_content(self, isolated_workspace):
        fo = self._ops()
        (isolated_workspace / "a.py").write_text("def foo():\n    pass\n")
        (isolated_workspace / "b.py").write_text("def bar():\n    pass\n")
        result = fo.grep_content("def foo")
        assert "a.py" in result
        assert "b.py" not in result
