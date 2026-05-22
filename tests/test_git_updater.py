"""Tests for utils.git_updater (mocked git)."""

from unittest.mock import patch

from utils import git_updater


def test_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTO_GIT_UPDATE", "0")
    result = git_updater.check_and_pull_updates()
    assert result["skipped"] is True
    assert result["updated"] is False


def test_up_to_date_after_fetch(monkeypatch):
    monkeypatch.setenv("AUTO_GIT_UPDATE", "1")
    monkeypatch.delenv("AUTO_GIT_UPDATE_ALLOW_DIRTY", raising=False)

    def fake_run(args, cwd=git_updater.ROOT):
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return 0, "true", ""
        if args == ["status", "--porcelain", "-uno"]:
            return 0, "", ""
        if args[:2] == ["fetch", "origin"]:
            return 0, "", ""
        if args == ["rev-parse", "--abbrev-ref", "@{u}"]:
            return 0, "origin/main", ""
        if args[:3] == ["rev-list", "--count", "HEAD..origin/main"]:
            return 0, "0", ""
        return 1, "", "unexpected"

    with patch.object(git_updater, "_run_git", side_effect=fake_run):
        result = git_updater.check_and_pull_updates(force=True)

    assert result["ok"] is True
    assert result["updated"] is False
    assert result["behind_before"] == 0


def test_pull_when_behind(monkeypatch):
    monkeypatch.setenv("AUTO_GIT_UPDATE", "1")

    calls = []

    def fake_run(args, cwd=git_updater.ROOT):
        calls.append(args)
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return 0, "true", ""
        if args == ["status", "--porcelain", "-uno"]:
            return 0, "", ""
        if args[:2] == ["fetch", "origin"]:
            return 0, "", ""
        if args == ["rev-parse", "--abbrev-ref", "@{u}"]:
            return 0, "origin/main", ""
        if args[:3] == ["rev-list", "--count", "HEAD..origin/main"]:
            return 0, "3", ""
        if args == ["merge", "--ff-only", "origin/main"]:
            return 0, "Updating abc..def\nFast-forward", ""
        return 1, "", "unexpected"

    with patch.object(git_updater, "_run_git", side_effect=fake_run):
        result = git_updater.check_and_pull_updates(force=True)

    assert result["updated"] is True
    assert result["commits_pulled"] == 3
    assert ["merge", "--ff-only", "origin/main"] in calls


def test_skip_on_dirty_tree(monkeypatch):
    monkeypatch.setenv("AUTO_GIT_UPDATE", "1")
    monkeypatch.setenv("AUTO_GIT_UPDATE_ALLOW_DIRTY", "0")

    def fake_run(args, cwd=git_updater.ROOT):
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return 0, "true", ""
        if args == ["status", "--porcelain", "-uno"]:
            return 0, " M requirements.txt", ""
        return 1, "", ""

    with patch.object(git_updater, "_run_git", side_effect=fake_run):
        result = git_updater.check_and_pull_updates(force=True)

    assert result["skipped"] is True
    assert result["updated"] is False
    assert "local changes" in result["reason"]
