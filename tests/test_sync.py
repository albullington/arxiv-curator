import subprocess
from pathlib import Path

import pytest

from arxiv_curator import sync


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _init_repo_with_identity(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "main"], cwd=path)
    _git(["config", "user.email", "test@example.com"], cwd=path)
    _git(["config", "user.name", "Test User"], cwd=path)


def _make_bare_remote_with_initial_commit(tmp_path: Path) -> Path:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    _git(["init", "--bare", "-b", "main"], cwd=bare)

    seed = tmp_path / "seed"
    _init_repo_with_identity(seed)
    (seed / "arxiv_curator.db").write_text("v1")
    _git(["add", "arxiv_curator.db"], cwd=seed)
    _git(["commit", "-m", "initial"], cwd=seed)
    _git(["remote", "add", "origin", str(bare)], cwd=seed)
    result = _git(["push", "-u", "origin", "main"], cwd=seed)
    assert result.returncode == 0, result.stderr
    return bare


def _clone(bare: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _git(["clone", str(bare), str(dest)], cwd=dest.parent)
    assert result.returncode == 0, result.stderr
    _git(["config", "user.email", "test@example.com"], cwd=dest)
    _git(["config", "user.name", "Test User"], cwd=dest)
    return dest


def test_sync_is_no_op_when_db_unchanged(tmp_path):
    bare = _make_bare_remote_with_initial_commit(tmp_path)
    local = _clone(bare, tmp_path / "local")
    before = _git(["rev-parse", "HEAD"], cwd=local).stdout

    result = sync.sync(local)

    assert result == "up-to-date"
    after = _git(["rev-parse", "HEAD"], cwd=local).stdout
    assert before == after


def test_sync_commits_and_pushes_when_db_changed(tmp_path):
    bare = _make_bare_remote_with_initial_commit(tmp_path)
    local = _clone(bare, tmp_path / "local")
    (local / "arxiv_curator.db").write_text("v2")

    result = sync.sync(local)

    assert result == "pushed local changes"
    log = _git(["log", "--oneline", "-1"], cwd=local).stdout
    assert "Update arxiv_curator.db" in log

    check = _clone(bare, tmp_path / "check")
    assert (check / "arxiv_curator.db").read_text() == "v2"


def test_pull_raises_clear_error_on_diverged_history(tmp_path):
    bare = _make_bare_remote_with_initial_commit(tmp_path)
    local_a = _clone(bare, tmp_path / "local_a")
    local_b = _clone(bare, tmp_path / "local_b")

    (local_b / "arxiv_curator.db").write_text("from-b")
    _git(["commit", "-am", "b's change"], cwd=local_b)
    push = _git(["push"], cwd=local_b)
    assert push.returncode == 0, push.stderr

    (local_a / "arxiv_curator.db").write_text("from-a")
    _git(["commit", "-am", "a's change"], cwd=local_a)

    with pytest.raises(sync.SyncError, match="diverged"):
        sync.pull(local_a)


def test_commit_and_push_raises_clear_error_when_rejected(tmp_path):
    bare = _make_bare_remote_with_initial_commit(tmp_path)
    local_a = _clone(bare, tmp_path / "local_a")
    local_b = _clone(bare, tmp_path / "local_b")

    (local_b / "arxiv_curator.db").write_text("from-b")
    _git(["commit", "-am", "b's change"], cwd=local_b)
    push = _git(["push"], cwd=local_b)
    assert push.returncode == 0, push.stderr

    (local_a / "arxiv_curator.db").write_text("from-a")

    with pytest.raises(sync.SyncError, match="rejected"):
        sync.commit_and_push(local_a)
