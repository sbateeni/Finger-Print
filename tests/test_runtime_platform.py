"""Cross-platform runtime defaults."""

from utils import runtime_platform


def test_live_reload_default_off(monkeypatch):
    monkeypatch.delenv("LIVE_RELOAD", raising=False)
    assert runtime_platform.live_reload_enabled() is False


def test_live_reload_explicit_on(monkeypatch):
    monkeypatch.setenv("LIVE_RELOAD", "1")
    assert runtime_platform.live_reload_enabled() is True


def test_database_url_no_backslashes():
    from database import DATABASE_URL

    assert DATABASE_URL.startswith("sqlite:///")
    path_part = DATABASE_URL[len("sqlite:///") :]
    assert "\\" not in path_part
