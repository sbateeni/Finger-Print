import json
from pathlib import Path

import pytest

from services.telegram_inbox import (
    INBOX_ROOT,
    format_paths_message,
    get_inbox_status,
    guess_image_suffix,
    load_pair_bytes,
    role_from_caption,
    save_inbox_image,
)


def test_guess_image_suffix():
    assert guess_image_suffix(b"\xff\xd8\xff") == ".jpg"
    assert guess_image_suffix(b"\x89PNG\r\n\x1a\n") == ".png"


def test_role_from_caption():
    assert role_from_caption("ref") == "reference"
    assert role_from_caption("مرجع") == "reference"
    assert role_from_caption("مقارنة") == "query"
    assert role_from_caption("hello") is None


def test_save_and_load_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("services.telegram_inbox.INBOX_ROOT", tmp_path / "inbox")
    chat = 999001
    ref_bytes = b"\xff\xd8\xff" + b"x" * 20
    qry_bytes = b"\x89PNG\r\n\x1a\n" + b"y" * 20

    save_inbox_image(chat, 1, "reference", ref_bytes)
    save_inbox_image(chat, 1, "query", qry_bytes)

    st = get_inbox_status(chat)
    assert st["has_reference"]
    assert st["has_query"]
    assert Path(st["reference_path"]).is_file()
    assert Path(st["query_path"]).is_file()

    rb, qb = load_pair_bytes(chat)
    assert rb == ref_bytes
    assert qb == qry_bytes

    msg = format_paths_message(chat)
    assert "test_pair_local.py" in msg or "scripts" in msg


def test_manifest_written(tmp_path, monkeypatch):
    monkeypatch.setattr("services.telegram_inbox.INBOX_ROOT", tmp_path / "inbox")
    chat = 42
    save_inbox_image(chat, 7, "reference", b"\xff\xd8\xffabc")
    manifest = json.loads((tmp_path / "inbox" / "42" / "latest" / "manifest.json").read_text())
    assert manifest["reference"]["role"] == "reference"
    assert manifest["user_id"] == 7
