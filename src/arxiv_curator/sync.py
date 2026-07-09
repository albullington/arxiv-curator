import subprocess
from pathlib import Path

DB_FILENAME = "arxiv_curator.db"


class SyncError(Exception):
    pass


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def pull(data_dir: Path) -> None:
    result = _run_git(["pull", "--ff-only"], cwd=data_dir)
    if result.returncode != 0:
        raise SyncError(
            f"git pull --ff-only failed (local and remote have diverged): {result.stderr.strip()}"
        )


def has_changes(data_dir: Path) -> bool:
    result = _run_git(["status", "--porcelain", "--", DB_FILENAME], cwd=data_dir)
    return bool(result.stdout.strip())


def commit_and_push(data_dir: Path) -> None:
    _run_git(["add", DB_FILENAME], cwd=data_dir)
    commit_result = _run_git(["commit", "-m", f"Update {DB_FILENAME}"], cwd=data_dir)
    if commit_result.returncode != 0:
        raise SyncError(f"git commit failed: {commit_result.stderr.strip()}")

    push_result = _run_git(["push"], cwd=data_dir)
    if push_result.returncode != 0:
        raise SyncError(
            f"git push rejected -- something else was pushed since your last sync. "
            f"Re-run sync: {push_result.stderr.strip()}"
        )


def sync(data_dir: Path) -> str:
    pull(data_dir)
    if not has_changes(data_dir):
        return "up-to-date"
    commit_and_push(data_dir)
    return "pushed local changes"
